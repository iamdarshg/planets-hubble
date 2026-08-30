"""Tests for the local Colab results receiver (upload/download daemon)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "colab_receiver", ROOT / "tools" / "colab_receiver.py"
)
receiver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receiver)


@pytest.fixture()
def server(tmp_path):
    upload_dir = tmp_path / "uploads"
    bundle_dir = tmp_path / "bundles"
    upload_dir.mkdir()
    bundle_dir.mkdir()
    (bundle_dir / "real_prepared_bundle.zip").write_bytes(b"ZIPDATA")
    instance = receiver.make_server(
        upload_dir=upload_dir,
        bundle_dir=bundle_dir,
        token="sekret",
        storage_cap_bytes=1_000_000,
        max_upload_bytes=1_000_000,
    )
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    thread.join(timeout=5)


def _base(instance) -> str:
    return f"http://127.0.0.1:{instance.server_address[1]}"


def request(method: str, url: str, data: bytes | None = None):
    req = urllib.request.Request(url, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_health_reports_service_state(server):
    status, body = request("GET", f"{_base(server)}/health")
    assert status == 200
    payload = json.loads(body)
    assert payload["ok"] is True
    assert payload["token_required"] is True


def test_upload_requires_valid_token(server):
    status, _ = request("POST", f"{_base(server)}/upload?name=a.pt&subdir=run1", data=b"x")
    assert status == 403
    status, _ = request(
        "POST", f"{_base(server)}/upload?token=wrong&name=a.pt&subdir=run1", data=b"x"
    )
    assert status == 403


def test_upload_stores_file_and_lists_it(server):
    payload = b"checkpoint-bytes"
    status, body = request(
        "POST",
        f"{_base(server)}/upload?token=sekret&name=ckpt.pt&subdir=run1",
        data=payload,
    )
    assert status == 200
    result = json.loads(body)
    assert result["bytes"] == len(payload)
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert (server.upload_dir / "run1" / "ckpt.pt").read_bytes() == payload

    status, body = request("GET", f"{_base(server)}/uploads?token=sekret")
    assert status == 200
    files = json.loads(body)["files"]
    assert {"subdir": "run1", "name": "ckpt.pt"} in [
        {"subdir": item["subdir"], "name": item["name"]} for item in files
    ]


def test_upload_rejects_path_traversal(server):
    status, _ = request(
        "POST",
        f"{_base(server)}/upload?token=sekret&name=..%2Fevil.pt&subdir=run1",
        data=b"x",
    )
    assert status == 400
    assert not (server.upload_dir.parent / "evil.pt").exists()
    status, _ = request(
        "POST",
        f"{_base(server)}/upload?token=sekret&name=ok.pt&subdir=..%2F..",
        data=b"x",
    )
    assert status == 400


def test_upload_rejects_oversize_payload(server):
    status, _ = request(
        "POST",
        f"{_base(server)}/upload?token=sekret&name=big.pt&subdir=run1",
        data=b"y" * 1_000_001,
    )
    assert status == 413


def test_upload_enforces_storage_cap(server):
    status, _ = request(
        "POST",
        f"{_base(server)}/upload?token=sekret&name=one.pt&subdir=run1",
        data=b"z" * 900_000,
    )
    assert status == 200
    status, _ = request(
        "POST",
        f"{_base(server)}/upload?token=sekret&name=two.pt&subdir=run1",
        data=b"z" * 900_000,
    )
    assert status == 507


def test_registered_upload_can_be_downloaded_back(server):
    payload = b"resume-me"
    request(
        "POST",
        f"{_base(server)}/upload?token=sekret&name=ckpt.pt&subdir=run1",
        data=payload,
    )
    status, body = request(
        "GET", f"{_base(server)}/files?token=sekret&name=run1%2Fckpt.pt"
    )
    assert status == 200
    assert body == payload


def test_bundle_download_requires_token(server):
    status, _ = request(
        "GET", f"{_base(server)}/bundle?name=real_prepared_bundle.zip"
    )
    assert status == 403


def test_bundle_download_serves_registered_zip(server):
    status, body = request(
        "GET",
        f"{_base(server)}/bundle?token=sekret&name=real_prepared_bundle.zip",
    )
    assert status == 200
    assert body == b"ZIPDATA"


def test_bundle_download_rejects_unknown_names(server):
    status, _ = request(
        "GET", f"{_base(server)}/bundle?token=sekret&name=secrets.tar"
    )
    assert status == 404


def test_unknown_routes_are_404(server):
    status, _ = request("GET", f"{_base(server)}/nope")
    assert status == 404
