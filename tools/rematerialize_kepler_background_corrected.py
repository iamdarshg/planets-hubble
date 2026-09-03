"""Re-materialize a Kepler corpus with the TPF sky background removed.

The original manifest and window timestamps are preserved exactly.  Only the
science and uncertainty planes are rebuilt from the retained raw TPFs, which
allows a preprocessing correction without changing host splits, labels, or
the permanently reserved holdout selection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_kepler_real_transit_corpus import _load_tpf, _write_example  # noqa: E402


def _records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _tpf_name(record: dict[str, object]) -> str:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"record has no provenance: {record.get('example_id')}")
    url = provenance.get("tpf_url")
    if not isinstance(url, str) or not url:
        raise ValueError(f"record has no TPF URL: {record.get('example_id')}")
    return url.rsplit("/", 1)[-1]


def _materialize_record(
    record: dict[str, object],
    *,
    input_root: Path,
    output_root: Path,
    times: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
    metadata: dict[str, object],
) -> dict[str, object]:
    source_path = input_root / str(record["path"])
    with np.load(source_path, allow_pickle=False) as arrays:
        window_times = np.asarray(arrays["time"], dtype=np.float64)
    if not np.isfinite(window_times).all():
        raise ValueError(f"window contains non-finite timestamps: {record.get('example_id')}")
    indices = np.asarray(
        [int(np.nanargmin(np.abs(times - value))) for value in window_times],
        dtype=np.int64,
    )
    if not np.allclose(times[indices], window_times, rtol=0.0, atol=2.0e-8):
        raise ValueError(f"window timestamps do not match raw TPF: {record.get('example_id')}")
    selected_flux = flux[indices]
    selected_err = flux_err[indices]
    selected_quality = quality[indices]
    output_path = output_root / str(record["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_example(
        output_path,
        times=window_times,
        flux=selected_flux,
        flux_err=selected_err,
        quality=selected_quality,
        label=int(record["label"]),
        metadata={**metadata, "tpf_url": str(record["provenance"]["tpf_url"])},
    )
    copy = dict(record)
    provenance = dict(record.get("provenance", {}))
    provenance["background_correction"] = "FLUX minus finite FLUX_BKG from retained raw TPF"
    provenance["background_uncertainty"] = "quadrature FLUX_ERR and FLUX_BKG_ERR when present"
    copy["provenance"] = provenance
    copy["path"] = os.path.relpath(output_path, output_root)
    return copy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--raw-tpf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_manifest = args.input_manifest.resolve()
    input_root = input_manifest.parent
    raw_dir = args.raw_tpf_dir.resolve()
    output_root = args.output_dir.resolve()
    if not input_manifest.is_file() or not raw_dir.is_dir():
        raise FileNotFoundError("input manifest and retained raw TPF directory are required")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_records = _records(input_manifest)
    by_tpf: dict[str, list[dict[str, object]]] = {}
    for record in source_records:
        by_tpf.setdefault(_tpf_name(record), []).append(record)

    corrected: list[dict[str, object]] = []
    for tpf_index, (name, group) in enumerate(sorted(by_tpf.items()), start=1):
        raw_path = raw_dir / name
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        times, flux, flux_err, quality, metadata = _load_tpf(raw_path)
        with __import__("astropy.io.fits", fromlist=["fits"]).open(raw_path, memmap=False) as hdul:
            table = hdul[1].data
            if "FLUX_BKG_ERR" in table.names:
                background_err = np.asarray(table["FLUX_BKG_ERR"], dtype=np.float32)
                flux_err = np.sqrt(
                    np.square(np.nan_to_num(flux_err, nan=0.0))
                    + np.square(np.nan_to_num(background_err, nan=0.0))
                ).astype(np.float32)
        metadata = {**metadata, "background_corrected": True}
        for record in group:
            corrected.append(
                _materialize_record(
                    record,
                    input_root=input_root,
                    output_root=output_root,
                    times=times,
                    flux=flux,
                    flux_err=flux_err,
                    quality=quality,
                    metadata=metadata,
                )
            )
        if tpf_index % 50 == 0 or tpf_index == len(by_tpf):
            print(json.dumps({"tpfs_processed": tpf_index, "tpfs_total": len(by_tpf), "records": len(corrected)}), flush=True)

    manifest_path = output_root / "corpus_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in source_records:
            replacement = next(item for item in corrected if item["example_id"] == record["example_id"])
            handle.write(json.dumps(replacement, sort_keys=True) + "\n")
    report = {
        "status": "complete",
        "input_manifest": str(input_manifest),
        "output_manifest": str(manifest_path),
        "raw_tpf_dir": str(raw_dir),
        "records": len(corrected),
        "unique_tpfs": len(by_tpf),
        "background_correction": "FLUX minus finite FLUX_BKG",
        "uncertainty_correction": "quadrature FLUX_ERR and FLUX_BKG_ERR when present",
        "labels_and_splits_preserved": True,
    }
    (output_root / "corpus_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
