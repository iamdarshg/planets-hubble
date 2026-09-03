"""Evaluate a checkpoint on the permanently reserved 100-record real holdout.

The holdout is the first 100 records in the Kepler corpus ``test`` split. It
is balanced in the acquired manifest and is never passed to a training or
simulator-calibration command. Keeping the selection here makes the gate
repeatable across synthetic calibration rounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from train_kepler_real import _load_model, _metrics, _records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aperture-fraction", type=float, default=1.0)
    parser.add_argument("--raw-tpf-dir", type=Path)
    parser.add_argument("--full-tpf-detrend", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not 0.0 < args.aperture_fraction <= 1.0:
        raise ValueError("aperture-fraction must be in (0, 1]")
    if args.full_tpf_detrend and args.raw_tpf_dir is None:
        raise ValueError("full-tpf-detrend requires --raw-tpf-dir")
    if args.full_tpf_detrend and not args.raw_tpf_dir.is_dir():
        raise FileNotFoundError(args.raw_tpf_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records = _records(args.manifest)
    test_records = [record for record in records if record.get("split") == "test"]
    holdout = test_records[:100]
    if len(holdout) != 100:
        raise ValueError(f"expected at least 100 test records, found {len(test_records)}")
    model = _load_model(args.checkpoint, device)
    metrics = _metrics(
        model,
        args.manifest.parent,
        holdout,
        device,
        args.batch_size,
        aperture_fraction=args.aperture_fraction,
        raw_tpf_dir=args.raw_tpf_dir,
        full_tpf_detrend=args.full_tpf_detrend,
    )
    report = {
        "status": "pass" if metrics["mean_bce_loss"] is not None and metrics["mean_bce_loss"] < 0.20 else "fail",
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "selection": "first 100 records in manifest test split; fixed and never trained on",
        "device": str(device),
        "batch_size": args.batch_size,
        "aperture_fraction": args.aperture_fraction,
        "full_tpf_detrend": args.full_tpf_detrend,
        "raw_tpf_dir": str(args.raw_tpf_dir) if args.raw_tpf_dir is not None else None,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
