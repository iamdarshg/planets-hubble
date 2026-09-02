"""Evaluate one checkpoint on independent synthetic and real domains."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "examples", ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_kepler_real as real  # noqa: E402
import train_synthetic_until_perfect as synthetic  # noqa: E402


def _metrics(labels: list[int], probabilities: list[float]) -> dict[str, object]:
    predictions = [int(value >= 0.5) for value in probabilities]
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    return {
        "samples": len(labels),
        "correct": correct,
        "accuracy": correct / len(labels) if labels else None,
        "mean_probability": sum(probabilities) / len(probabilities) if probabilities else None,
        "true_positive": sum(prediction == label == 1 for prediction, label in zip(predictions, labels)),
        "true_negative": sum(prediction == label == 0 for prediction, label in zip(predictions, labels)),
        "false_positive": sum(prediction == 1 and label == 0 for prediction, label in zip(predictions, labels)),
        "false_negative": sum(prediction == 0 and label == 1 for prediction, label in zip(predictions, labels)),
    }


def _real_eval(checkpoint: Path, manifest: Path, batch_size: int) -> dict[str, object]:
    device = torch.device("cpu")
    model = real._load_model(checkpoint, device)
    root = manifest.parent
    records = [record for record in real._records(manifest) if record.get("split") == "test"]
    labels: list[int] = []
    probabilities: list[float] = []
    model.eval()
    with torch.inference_mode():
        for batch_records in real._batch_records(records, batch_size, seed=0, shuffle=False):
            batch = real._make_batch(root, batch_records, device)
            output = model(batch)
            probabilities.extend(float(value) for value in output["global_event_logits"].float().sigmoid().reshape(-1).cpu())
            labels.extend(int(record["label"]) for record in batch_records)
            del batch, output
    result = _metrics(labels, probabilities)
    result["manifest"] = str(manifest)
    result["split"] = "test"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-manifest", type=Path, required=True)
    parser.add_argument("--synthetic-pairs", type=int, default=256)
    parser.add_argument("--synthetic-start-index", type=int, default=100000)
    parser.add_argument("--batch-pairs", type=int, default=64)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.real_manifest.is_file():
        raise FileNotFoundError("checkpoint and real manifest are required")
    if args.synthetic_pairs < 1 or args.batch_pairs < 1:
        raise ValueError("synthetic-pairs and batch-pairs must be positive")

    device = torch.device("cpu")
    model = synthetic._load_model(args.checkpoint, device)
    config = synthetic._synthetic_config(seed=23)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    synthetic_result = synthetic._evaluate(
        model,
        config,
        args.synthetic_start_index,
        args.synthetic_pairs,
        args.batch_pairs,
        device,
        args.cache_dir,
    )
    real_result = _real_eval(args.checkpoint, args.real_manifest, args.batch_pairs)
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_modified": False,
        "device": "cpu",
        "synthetic": synthetic_result,
        "real": real_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
