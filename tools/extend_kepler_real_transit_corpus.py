"""Extract additional real Kepler transit/control windows from retained TPFs.

The initial corpus deliberately kept one positive/control pair per confirmed
host.  This extension reuses the retained real target-pixel files and the
published DR25 ephemerides to select different observed transit epochs and
their off-transit controls.  It never injects a synthetic signal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_kepler_real_transit_corpus import (  # noqa: E402
    BASE_URL,
    _catalog_rows,
    _load_tpf,
    _sha256,
    _split_for_host,
    _window,
    _window_metrics,
    _write_example,
)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _tpf_url(kepid: int, name: str) -> str:
    padded = f"{kepid:09d}"
    return f"{BASE_URL}{padded[:4]}/{padded}/{name}"


def _centres(
    *,
    times: np.ndarray,
    period: float,
    epoch: float,
    duration_hours: float,
    frames: int,
) -> list[float]:
    finite = times[np.isfinite(times)]
    if finite.size == 0:
        return []
    first = math.floor((float(finite.min()) - epoch) / period) - 1
    last = math.ceil((float(finite.max()) - epoch) / period) + 1
    return [epoch + index * period for index in range(first, last + 1)]


def _near_existing(center: float, existing: list[float], tolerance_days: float) -> bool:
    return any(abs(center - previous) <= tolerance_days for previous in existing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--additional-pairs", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument(
        "--existing-window-tolerance-days",
        type=float,
        default=0.25,
        help="skip windows close to already materialized windows",
    )
    args = parser.parse_args()
    if args.additional_pairs < 1 or args.frames < 4 or args.existing_window_tolerance_days < 0:
        raise ValueError("additional-pairs and frames must be positive; tolerance cannot be negative")

    primary_dir = args.primary_dir.resolve()
    primary_manifest = primary_dir / "corpus_manifest.jsonl"
    raw_dir = primary_dir / "raw-tpf"
    if not primary_manifest.is_file() or not raw_dir.is_dir():
        raise FileNotFoundError("primary manifest and retained raw-tpf directory are required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = args.output_dir / "examples"
    examples_dir.mkdir()

    primary_records = _read_jsonl(primary_manifest)
    primary_by_host: dict[int, dict[str, object]] = {}
    existing_centres: dict[int, list[float]] = {}
    for record in primary_records:
        kepid = int(record["kepid"])
        primary_by_host.setdefault(kepid, record)
        existing_centres.setdefault(kepid, []).append(float(record["center_bkjd"]))

    catalog_by_host = {int(row["kepid"]): row for row in _catalog_rows(args.catalog)}
    raw_files: list[tuple[int, Path]] = []
    for kepid in sorted(primary_by_host):
        raw_files.extend((kepid, path) for path in sorted(raw_dir.glob(f"kplr{kepid:09d}-*.fits.gz")))

    output_manifest = args.output_dir / "extension_manifest.jsonl"
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    seen_windows: set[tuple[int, str, int]] = set()
    pair_index = 0

    # Deterministic ordering makes reruns and audit comparisons reproducible.
    for kepid, tpf_path in raw_files:
        if pair_index >= args.additional_pairs:
            break
        row = catalog_by_host.get(kepid)
        if row is None:
            failures.append({"kepid": str(kepid), "file": tpf_path.name, "error": "no catalog row"})
            continue
        try:
            period = float(row["koi_period"])
            epoch = float(row["koi_time0bk"])
            duration_hours = float(row["koi_duration"])
            times, flux, flux_err, quality, tpf_metadata = _load_tpf(tpf_path)
            if not np.isfinite(times).any():
                raise ValueError("TPF contains no finite cadence times")
            for positive_center in _centres(
                times=times,
                period=period,
                epoch=epoch,
                duration_hours=duration_hours,
                frames=args.frames,
            ):
                if pair_index >= args.additional_pairs:
                    break
                if _near_existing(positive_center, existing_centres[kepid], args.existing_window_tolerance_days):
                    continue
                negative = None
                negative_center = None
                for possible_center in (positive_center + period / 2.0, positive_center - period / 2.0):
                    negative = _window(
                        times,
                        flux,
                        flux_err,
                        quality,
                        possible_center,
                        duration_hours,
                        frames=args.frames,
                    )
                    if negative is not None and not _near_existing(
                        possible_center, existing_centres[kepid], args.existing_window_tolerance_days
                    ):
                        negative_center = possible_center
                        break
                positive = _window(
                    times,
                    flux,
                    flux_err,
                    quality,
                    positive_center,
                    duration_hours,
                    frames=args.frames,
                )
                if positive is None or negative is None or negative_center is None:
                    continue
                signature = (kepid, tpf_path.name, int(round(positive_center * 1_000_000)))
                if signature in seen_windows:
                    continue
                seen_windows.add(signature)
                tpf_url = _tpf_url(kepid, tpf_path.name)
                base_metadata = {
                    "catalog": "MAST Kepler KOI DR25 tabdelimited 2017-03-27",
                    "catalog_row": row,
                    "tpf_url": tpf_url,
                    "local_tpf": os.path.relpath(tpf_path, args.output_dir),
                    "tpf_metadata": tpf_metadata,
                    "published_epoch_bkjd": epoch,
                    "published_period_days": period,
                    "published_duration_hours": duration_hours,
                    "selected_transit_center_bkjd": positive_center,
                    "control_center_bkjd": negative_center,
                    "label_source": "published confirmed Kepler planet ephemeris; no synthetic injection",
                    "positive_window_metrics": _window_metrics(positive),
                    "control_window_metrics": _window_metrics(negative),
                    "extension_source": "additional observed epoch or quarter from retained Kepler TPF",
                }
                pair_name = f"extended-{pair_index:04d}"
                for suffix, payload, label, center in (
                    ("positive", positive, 1, positive_center),
                    ("control", negative, 0, negative_center),
                ):
                    filename = f"{pair_name}-{suffix}.npz"
                    out_path = examples_dir / filename
                    _write_example(
                        out_path,
                        times=payload[0],
                        flux=payload[1],
                        flux_err=payload[2],
                        quality=payload[3],
                        label=label,
                        metadata={**tpf_metadata, "kepid": kepid, "tpf_url": tpf_url},
                    )
                    records.append(
                        {
                            "example_id": f"kepler-dr25-{pair_name}-{suffix}",
                            "path": str(out_path.relative_to(args.output_dir)),
                            "label": label,
                            "label_name": "confirmed_transit" if label else "ephemeris_off_transit_control",
                            "target_name": row["kepler_name"],
                            "kepid": kepid,
                            "kepoi_name": row["kepoi_name"],
                            "split": _split_for_host(kepid),
                            "center_bkjd": center,
                            "provenance": base_metadata,
                        }
                    )
                existing_centres[kepid].extend([positive_center, float(negative_center)])
                pair_index += 1
                if pair_index % 25 == 0:
                    print(json.dumps({"additional_pairs": pair_index, "last_kepid": kepid}), flush=True)
        except Exception as exc:
            failures.append({"kepid": str(kepid), "file": tpf_path.name, "error": str(exc)})

    with output_manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    combined_dir = args.output_dir.parent / "kepler-expanded-4000"
    combined_dir.mkdir(parents=True, exist_ok=True)
    combined_manifest = combined_dir / "corpus_manifest.jsonl"
    combined_records: list[dict[str, object]] = []
    for record in primary_records + records:
        copy = dict(record)
        original_root = primary_dir if record in primary_records else args.output_dir
        source_path = (original_root / str(record["path"])).resolve()
        copy["path"] = os.path.relpath(source_path, combined_dir)
        combined_records.append(copy)
    with combined_manifest.open("w", encoding="utf-8") as handle:
        for record in combined_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    counts = {}
    for record in combined_records:
        split = str(record["split"])
        counts.setdefault(split, {"records": 0, "positive": 0, "control": 0, "unique_hosts": set()})
        counts[split]["records"] += 1
        counts[split]["positive" if int(record["label"]) else "control"] += 1
        counts[split]["unique_hosts"].add(int(record["kepid"]))
    report = {
        "status": "complete" if pair_index >= args.additional_pairs else "insufficient_examples",
        "additional_pairs": pair_index,
        "additional_examples": len(records),
        "total_pairs": len(combined_records) // 2,
        "total_examples": len(combined_records),
        "primary_manifest": str(primary_manifest),
        "extension_manifest": str(output_manifest),
        "combined_manifest": str(combined_manifest),
        "catalog": str(args.catalog),
        "catalog_sha256": _sha256(args.catalog),
        "raw_tpf_dir": str(raw_dir),
        "raw_tpf_files_considered": len(raw_files),
        "failures": failures,
        "split_counts": {
            split: {
                **{key: value for key, value in detail.items() if key != "unique_hosts"},
                "unique_hosts": len(detail["unique_hosts"]),
            }
            for split, detail in sorted(counts.items())
        },
        "frames": args.frames,
        "existing_window_tolerance_days": args.existing_window_tolerance_days,
        "label_source": "published confirmed Kepler ephemerides; additional observed TPF epochs; no synthetic injection",
    }
    (args.output_dir / "extension_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (combined_dir / "corpus_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "additional_pairs", "additional_examples", "total_examples")}, sort_keys=True))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
