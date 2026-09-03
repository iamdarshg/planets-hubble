"""Build longer, background-corrected Kepler windows from retained TPFs.

The input manifest supplies the published-transit/control centers and the
fixed host split.  This tool changes only the cadence context and output paths;
labels, ordering, and provenance remain tied to the original records.
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

from build_kepler_real_transit_corpus import _load_tpf, _window, _write_example  # noqa: E402


def _records(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _tpf_name(record: dict[str, object]) -> str:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("tpf_url"), str):
        raise ValueError(f"record has no TPF URL: {record.get('example_id')}")
    return str(provenance["tpf_url"]).rsplit("/", 1)[-1]


def _background_corrected_errors(raw_path: Path, flux_err: np.ndarray) -> np.ndarray:
    from astropy.io import fits

    with fits.open(raw_path, memmap=False) as hdul:
        table = hdul[1].data
        if "FLUX_BKG_ERR" not in table.names:
            return flux_err
        background_err = np.asarray(table["FLUX_BKG_ERR"], dtype=np.float32)
    return np.sqrt(
        np.square(np.nan_to_num(flux_err, nan=0.0))
        + np.square(np.nan_to_num(background_err, nan=0.0))
    ).astype(np.float32)


def _materialize(
    record: dict[str, object],
    *,
    input_root: Path,
    output_root: Path,
    times: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
    metadata: dict[str, object],
    frames: int,
    ordinal: int,
) -> dict[str, object]:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"record has no provenance: {record.get('example_id')}")
    center = float(record["center_bkjd"])
    duration = float(provenance["published_duration_hours"])
    window = None
    actual_frames = frames
    for candidate_frames in dict.fromkeys((frames, 48, 32, 16)):
        if candidate_frames > frames:
            continue
        window = _window(times, flux, flux_err, quality, center, duration, frames=candidate_frames)
        if window is not None:
            actual_frames = candidate_frames
            break
    if window is None:
        raise ValueError(f"TPF cannot provide {frames} finite cadences around {center}: {record.get('example_id')}")
    window_times, window_flux, window_err, window_quality = window
    filename = f"{ordinal:04d}-{str(record['label'])}.npz"
    output_path = output_root / "examples" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_example(
        output_path,
        times=window_times,
        flux=window_flux,
        flux_err=window_err,
        quality=window_quality,
        label=int(record["label"]),
        metadata={**metadata, "tpf_url": str(provenance["tpf_url"])},
    )
    copy = dict(record)
    copy["path"] = os.path.relpath(output_path, output_root)
    updated_provenance = dict(provenance)
    updated_provenance["background_correction"] = "FLUX minus finite FLUX_BKG from retained raw TPF"
    updated_provenance["background_uncertainty"] = "quadrature FLUX_ERR and FLUX_BKG_ERR when present"
    updated_provenance["window_frames"] = actual_frames
    updated_provenance["window_context_days"] = float(window_times[-1] - window_times[0])
    copy["provenance"] = updated_provenance
    return copy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--raw-tpf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()
    if args.frames < 16:
        raise ValueError("frames must be at least 16")
    input_manifest = args.input_manifest.resolve()
    raw_dir = args.raw_tpf_dir.resolve()
    output_root = args.output_dir.resolve()
    if not input_manifest.is_file() or not raw_dir.is_dir():
        raise FileNotFoundError("input manifest and retained raw TPF directory are required")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_records = _records(input_manifest)
    by_tpf: dict[str, list[tuple[int, dict[str, object]]]] = {}
    for ordinal, record in enumerate(source_records):
        by_tpf.setdefault(_tpf_name(record), []).append((ordinal, record))

    rebuilt: dict[str, dict[str, object]] = {}
    for index, (name, group) in enumerate(sorted(by_tpf.items()), start=1):
        raw_path = raw_dir / name
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        times, flux, flux_err, quality, metadata = _load_tpf(raw_path)
        flux_err = _background_corrected_errors(raw_path, flux_err)
        metadata = {**metadata, "background_corrected": True}
        for ordinal, record in group:
            rebuilt[str(record["example_id"])] = _materialize(
                record,
                input_root=input_manifest.parent,
                output_root=output_root,
                times=times,
                flux=flux,
                flux_err=flux_err,
                quality=quality,
                metadata=metadata,
                frames=args.frames,
                ordinal=ordinal,
            )
        if index % 50 == 0 or index == len(by_tpf):
            print(json.dumps({"tpfs_processed": index, "tpfs_total": len(by_tpf), "records": len(rebuilt)}), flush=True)

    manifest_path = output_root / "corpus_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in source_records:
            handle.write(json.dumps(rebuilt[str(record["example_id"])], sort_keys=True) + "\n")
    report = {
        "status": "complete",
        "input_manifest": str(input_manifest),
        "output_manifest": str(manifest_path),
        "raw_tpf_dir": str(raw_dir),
        "records": len(source_records),
        "unique_tpfs": len(by_tpf),
        "requested_frames": args.frames,
        "actual_frame_counts": sorted({int(record["provenance"]["window_frames"]) for record in rebuilt.values()}),
        "background_correction": "FLUX minus finite FLUX_BKG",
        "uncertainty_correction": "quadrature FLUX_ERR and FLUX_BKG_ERR when present",
        "labels_splits_and_order_preserved": True,
    }
    (output_root / "corpus_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
