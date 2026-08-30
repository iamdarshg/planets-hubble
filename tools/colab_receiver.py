"""Small local receiver daemon for Colab-run results.

The Colab notebook has no inbound connection to this machine, so this process
is exposed through a free cloudflared quick tunnel.  It accepts only:

* POST /upload - write one uploaded file into the upload directory;
* GET /uploads - list the registry of accepted uploads;
* GET /files - read back a registered upload (used to resume Colab runs);
* GET /bundle - serve a pre-registered bundle (the prepared real dataset);
* GET /health - status used by the notebook to verify the endpoint.

The receiver deliberately has no arbitrary file read, no directory listing of
the host, no shell execution, and no delete endpoint.  Filenames and subdirs
are strictly validated, uploads are size-capped and storage-capped, and every
write is atomic.  The token is not a real security boundary (it ships to the
public notebook), so keep the upload directory disposable and never point the
tunnel at anything else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import uuid4


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SAFE_SUBDIR = re.compile(r"^[A-Za-z0-9_-]+$")
REGISTRY_NAME = "uploads.json"
DEFAULT_TOKEN_PATH = Path(__file__).resolve().with_name("colab_receiver.token")
DEFAULT_UPLOAD_DIR = Path("artifacts/colab-uploads/uploads")
DEFAULT_BUNDLE_DIR = Path("artifacts/colab-uploads/bundles")
DEFAULT_STORAGE_CAP_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
DRAIN_LIMIT_BYTES = 8 * 1024 * 1024


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_registry(upload_dir: Path) -> list[dict[str, object]]:
    path = upload_dir / REGISTRY_NAME
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _write_registry(upload_dir: Path, entries: list[dict[str, object]]) -> None:
    path = upload_dir / REGISTRY_NAME
    temporary = path.with_name(f".{REGISTRY_NAME}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_or_create_token(token_path: Path) -> str:
    token_path = Path(token_path)
    if token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_hex(32)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token + "\n", encoding="utf-8")
    return token


class ColabReceiverServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address,
        *,
        upload_dir: Path,
        bundle_dir: Path,
        token: str,
        storage_cap_bytes: int,
        max_upload_bytes: int,
    ) -> None:
        super().__init__(address, ColabReceiverHandler)
        self.upload_dir = Path(upload_dir)
        self.bundle_dir = Path(bundle_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.token = token
        self.storage_cap_bytes = storage_cap_bytes
        self.max_upload_bytes = max_upload_bytes
        self._lock = threading.RLock()

    @property
    def registry(self) -> list[dict[str, object]]:
        return _read_registry(self.upload_dir)

    @property
    def storage_bytes(self) -> int:
        return sum(int(entry.get("bytes", 0)) for entry in self.registry)

    def register(self, subdir: str, name: str, byte_count: int, sha256: str) -> None:
        entries = self.registry
        entries.append(
            {
                "subdir": subdir,
                "name": name,
                "bytes": byte_count,
                "sha256": sha256,
            }
        )
        with self._lock:
            _write_registry(self.upload_dir, entries)


class ColabReceiverHandler(BaseHTTPRequestHandler):
    server: ColabReceiverServer

    def log_message(self, fmt: str, *args) -> None:  # keep the console quiet
        return

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _query(self) -> dict[str, list[str]]:
        return parse_qs(urlparse(self.path).query)

    def _valid_token(self, query: dict[str, list[str]]) -> bool:
        supplied = (query.get("token") or [""])[0]
        return bool(supplied) and secrets.compare_digest(supplied, self.server.token)

    def _valid_name(self, name: str | None) -> bool:
        return bool(name) and SAFE_NAME.fullmatch(name) is not None

    def _valid_subdir(self, subdir: str | None) -> bool:
        return bool(subdir) and SAFE_SUBDIR.fullmatch(subdir) is not None

    def _drain_request_body(self, byte_count: int) -> None:
        """Consume a bounded prefix of an unaccepted body before responding."""

        amount = min(byte_count, DRAIN_LIMIT_BYTES)
        if amount > 0:
            self.rfile.read(amount)

    def _resolve_registered(self, subdir: str, name: str) -> Path | None:
        base = self.server.upload_dir.resolve()
        candidate = (base / subdir / name).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = self._query()
        path = parsed.path
        if path == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "planets-hubble-colab-receiver",
                    "token_required": True,
                    "uploads": len(self.server.registry),
                    "storage_bytes": self.server.storage_bytes,
                    "storage_cap_bytes": self.server.storage_cap_bytes,
                },
            )
            return
        if path not in {"/uploads", "/files", "/bundle"}:
            self._send_json(404, {"error": "not found"})
            return
        if not self._valid_token(query):
            self._send_json(403, {"error": "invalid token"})
            return
        if path == "/uploads":
            self._send_json(200, {"files": self.server.registry})
            return
        if path == "/files":
            relative = (query.get("name") or [""])[0]
            if "/" not in relative:
                self._send_json(400, {"error": "name must be subdir/file"})
                return
            subdir, name = relative.split("/", 1)
            if not self._valid_subdir(subdir) or not self._valid_name(name):
                self._send_json(400, {"error": "invalid name"})
                return
            registered = {
                (str(entry.get("subdir")), str(entry.get("name")))
                for entry in self.server.registry
            }
            if (subdir, name) not in registered:
                self._send_json(404, {"error": "upload not found"})
                return
            path_on_disk = self._resolve_registered(subdir, name)
            if path_on_disk is None:
                self._send_json(404, {"error": "upload file missing"})
                return
            self._send_bytes(200, path_on_disk.read_bytes(), "application/octet-stream")
            return
        if path == "/bundle":
            name = (query.get("name") or [""])[0]
            if not self._valid_name(name):
                self._send_json(400, {"error": "invalid bundle name"})
                return
            path_on_disk = self.server.bundle_dir / name
            if not path_on_disk.is_file():
                self._send_json(404, {"error": "bundle not found"})
                return
            self._send_bytes(200, path_on_disk.read_bytes(), "application/octet-stream")
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/upload":
            self._send_json(404, {"error": "not found"})
            return
        query = self._query()
        if not self._valid_token(query):
            self._send_json(403, {"error": "invalid token"})
            return
        name = (query.get("name") or [""])[0]
        subdir = (query.get("subdir") or [""])[0]
        if not self._valid_name(name) or not self._valid_subdir(subdir):
            self._send_json(400, {"error": "invalid name or subdir"})
            return
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_json(411, {"error": "Content-Length required"})
            return
        try:
            byte_count = int(raw_length)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if byte_count > self.server.max_upload_bytes:
            self._drain_request_body(byte_count)
            self._send_json(413, {"error": "upload exceeds size cap"})
            return
        with self.server._lock:
            if self.server.storage_bytes + byte_count > self.server.storage_cap_bytes:
                self._drain_request_body(byte_count)
                self._send_json(507, {"error": "upload storage cap exceeded"})
                return
            body = self.rfile.read(byte_count)
            if len(body) != byte_count:
                self._send_json(400, {"error": "truncated upload"})
                return
            destination_dir = self.server.upload_dir / subdir
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / name
            temporary = destination_dir / f".{name}.{uuid4().hex}.tmp"
            try:
                with temporary.open("xb") as handle:
                    handle.write(body)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
            digest = _sha256_hex(body)
            self.server.register(subdir, name, byte_count, digest)
        self._send_json(
            200,
            {
                "ok": True,
                "name": f"{subdir}/{name}",
                "bytes": byte_count,
                "sha256": digest,
            },
        )


def make_server(
    *,
    upload_dir: str | Path,
    bundle_dir: str | Path,
    token: str,
    storage_cap_bytes: int = DEFAULT_STORAGE_CAP_BYTES,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    port: int = 0,
) -> ColabReceiverServer:
    return ColabReceiverServer(
        ("127.0.0.1", port),
        upload_dir=upload_dir,
        bundle_dir=bundle_dir,
        token=token,
        storage_cap_bytes=storage_cap_bytes,
        max_upload_bytes=max_upload_bytes,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_PATH)
    parser.add_argument(
        "--storage-cap-bytes",
        type=int,
        default=int(
            os.environ.get(
                "PLANETS_HUBBLE_RECEIVER_STORAGE_CAP_BYTES", DEFAULT_STORAGE_CAP_BYTES
            )
        ),
    )
    parser.add_argument(
        "--max-upload-bytes",
        type=int,
        default=int(
            os.environ.get(
                "PLANETS_HUBBLE_RECEIVER_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES
            )
        ),
    )
    args = parser.parse_args()
    token = os.environ.get("PLANETS_HUBBLE_RECEIVER_TOKEN") or load_or_create_token(
        args.token_file
    )
    server = make_server(
        upload_dir=args.upload_dir,
        bundle_dir=args.bundle_dir,
        token=token,
        storage_cap_bytes=args.storage_cap_bytes,
        max_upload_bytes=args.max_upload_bytes,
        port=args.port,
    )
    print(
        json.dumps(
            {
                "status": "listening",
                "host": args.host,
                "port": server.server_address[1],
                "upload_dir": str(server.upload_dir),
                "bundle_dir": str(server.bundle_dir),
                "token_file": str(args.token_file),
                "storage_cap_bytes": server.storage_cap_bytes,
                "max_upload_bytes": server.max_upload_bytes,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
