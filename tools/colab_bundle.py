"""Build the prepared real-data bundle the Colab notebook downloads.

Only the *_prepared directories are bundled (calibrated crops, DQ, and
manifests).  Raw FITS files, CRDS assets, and PSF libraries stay on the host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def build_bundle(source_roots: list[Path], output: Path) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"version": 1, "files": []}
    file_count = 0
    total_bytes = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, sort_keys=True),
        )
        for root in source_roots:
            root = Path(root).resolve()
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(root.parent)
                archive.write(path, str(relative))
                entry = {
                    "path": str(relative),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                manifest["files"].append(entry)
                file_count += 1
                total_bytes += int(entry["bytes"])
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": digest,
        "source_files": file_count,
        "source_bytes": total_bytes,
        "source_roots": [str(Path(root).resolve()) for root in source_roots],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-roots",
        type=Path,
        nargs="*",
        default=None,
        help="prepared directories to bundle; default: data/real/*_prepared",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/colab-uploads/bundles/real_prepared_bundle.zip"),
    )
    args = parser.parse_args()
    if args.source_roots is None:
        args.source_roots = sorted(Path("data/real").glob("*_prepared"))
    summary = build_bundle(args.source_roots, args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
