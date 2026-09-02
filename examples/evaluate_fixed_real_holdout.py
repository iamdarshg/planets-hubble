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
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    records = _records(args.manifest)
    test_records = [record for record in records if record.get("split") == "test"]
    holdout = test_records[:100]
    if len(holdout) != 100:
        raise ValueError(f"expected at least 100 test records, found {len(test_records)}")
    model = _load_model(args.checkpoint, device)
    metrics = _metrics(model, args.manifest.parent, holdout, device, args.batch_size)
    report = {
        "status": "pass" if metrics["mean_bce_loss"] is not None and metrics["mean_bce_loss"] < 0.20 else "fail",
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "selection": "first 100 records in manifest test split; fixed and never trained on",
        "device": str(device),
        "batch_size": args.batch_size,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
