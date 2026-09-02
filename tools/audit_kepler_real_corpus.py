"""Audit observed Kepler windows for split, provenance, and transit resolution.

The catalog label is an ephemeris label.  This command deliberately does not
rewrite labels or promote a photometric heuristic to ground truth.  It reports
whether each saved window contains enough off-center cadences to estimate a
local baseline and records a conservative center-dip diagnostic for deciding
which examples are suitable for supervised training versus weakly labelled
evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _records(manifest: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"empty manifest: {manifest}")
    return records


def _example_audit(root: Path, record: dict[str, object]) -> dict[str, object]:
    with np.load(root / str(record["path"]), allow_pickle=False) as arrays:
        times = np.asarray(arrays["time"], dtype=np.float64)
        science = np.asarray(arrays["science"], dtype=np.float64)
        finite = np.asarray(arrays["finite"], dtype=bool)
        quality = np.asarray(arrays["quality"], dtype=np.int32)
    aperture = np.nansum(np.where(finite, science, 0.0), axis=(1, 2))
    valid = np.isfinite(times) & np.isfinite(aperture) & (aperture > 0.0)
    center = float(record["center_bkjd"])
    duration_days = max(float(record["provenance"]["published_duration_hours"]) / 24.0, 0.0204)
    inner = valid & (np.abs(times - center) <= duration_days / 2.0)
    outer = valid & (np.abs(times - center) >= duration_days)
    finite_depth = bool(inner.any() and outer.any())
    if finite_depth:
        baseline = float(np.median(aperture[outer]))
        center_median = float(np.median(aperture[inner]))
        center_minimum = float(np.min(aperture[inner]))
        depth_median = (baseline - center_median) / max(abs(baseline), 1.0e-12)
        depth_minimum = (baseline - center_minimum) / max(abs(baseline), 1.0e-12)
    else:
        depth_median = None
        depth_minimum = None
    nearest_offset = float(np.min(np.abs(times[valid] - center))) if valid.any() else None
    return {
        "example_id": record["example_id"],
        "label": int(record["label"]),
        "split": record["split"],
        "kepid": int(record["kepid"]),
        "target_name": record["target_name"],
        "finite_cadences": int(valid.sum()),
        "inner_cadences": int(inner.sum()),
        "outer_cadences": int(outer.sum()),
        "quality_nonzero_cadences": int(np.count_nonzero(quality)),
        "nearest_cadence_offset_days": nearest_offset,
        "has_local_baseline": finite_depth,
        "center_depth_median": depth_median,
        "center_depth_minimum": depth_minimum,
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {
        "examples": len(rows),
        "labels": dict(sorted(Counter(int(row["label"]) for row in rows).items())),
        "hosts": len({int(row["kepid"]) for row in rows}),
        "with_local_baseline": sum(bool(row["has_local_baseline"]) for row in rows),
    }
    depths = np.asarray(
        [float(row["center_depth_median"]) for row in rows if row["center_depth_median"] is not None],
        dtype=np.float64,
    )
    if depths.size:
        result.update(
            median_center_depth=float(np.median(depths)),
            mean_center_depth=float(np.mean(depths)),
            fraction_center_dip_gt_100ppm=float(np.mean(depths > 1.0e-4)),
        )
    else:
        result.update(median_center_depth=None, mean_center_depth=None, fraction_center_dip_gt_100ppm=None)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = _records(args.manifest)
    root = args.manifest.parent
    audited = [_example_audit(root, record) for record in records]
    split_hosts = {
        split: sorted({int(row["kepid"]) for row in audited if row["split"] == split})
        for split in ("train", "validation", "test")
    }
    overlaps = {
        f"{left}_{right}": sorted(set(split_hosts[left]) & set(split_hosts[right]))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    summary = {
        "status": "complete",
        "manifest": str(args.manifest),
        "dataset": _summary(audited),
        "splits": {
            split: _summary([row for row in audited if row["split"] == split])
            for split in ("train", "validation", "test")
        },
        "host_overlap": {name: len(hosts) for name, hosts in overlaps.items()},
        "quality_flags": {
            "no_local_baseline": sum(not bool(row["has_local_baseline"]) for row in audited),
            "positive_no_local_baseline": sum(
                int(row["label"]) == 1 and not bool(row["has_local_baseline"]) for row in audited
            ),
            "control_no_local_baseline": sum(
                int(row["label"]) == 0 and not bool(row["has_local_baseline"]) for row in audited
            ),
        },
        "examples": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "dataset": summary["dataset"], "quality_flags": summary["quality_flags"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
