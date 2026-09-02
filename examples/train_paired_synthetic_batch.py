"""Train on a locally rendered synthetic pair stream with batch size two.

The regular lifecycle splits each null/injected pair into two GPU passes to
protect the tighter historical host limit.  This runner is for machines with
the explicitly raised RSS budget: it keeps both views together, so each pair
uses one forward/backward/optimizer pass while preserving the shared nuisance
realization.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import research_config  # noqa: E402
from synthetic import SyntheticConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=2048)
    parser.add_argument("--pair-count", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--rss-cap-bytes", type=int, default=1_879_048_192)
    parser.add_argument("--storage-cap-bytes", type=int, default=5 * 1024**3)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument(
        "--cudnn-off",
        action="store_true",
        help="disable cuDNN workspace allocation to reduce host RSS",
    )
    args = parser.parse_args()
    if args.start_index < 0 or args.pair_count < 1:
        raise ValueError("start-index must be non-negative and pair-count must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if args.rss_cap_bytes < 1 or args.storage_cap_bytes < 1:
        raise ValueError("resource caps must be positive")
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    rows = _validate_generation_manifest(
        args.generation_dir,
        start_index=args.start_index,
        pair_count=args.pair_count,
    )

    # These imports read the resource-cap environment at module import time;
    # set the requested cap before loading the training harness.
    os.environ["PLANETS_HUBBLE_RSS_CAP_BYTES"] = str(args.rss_cap_bytes)
    os.environ["PLANETS_HUBBLE_STORAGE_CAP_BYTES"] = str(args.storage_cap_bytes)
    from training import (  # noqa: E402
        AstroMambaHTrainingAdapter,
        BoundedTrainer,
        TrainingConfig,
        iter_paired_synthetic_training_batches,
        resolve_device,
    )
    from training.harness import default_loss_fn  # noqa: E402

    device = resolve_device(args.device)
    config = replace(research_config(), decode_heatmaps=False)
    construction_context = torch.device(device) if device.type == "cuda" else None
    if construction_context is None:
        model = AstroMambaHTrainingAdapter(config=config)
    else:
        with construction_context:
            model = AstroMambaHTrainingAdapter(config=config)
    model = model.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)

    trainer = BoundedTrainer(
        model,
        config=TrainingConfig(
            device=device,
            max_batches_per_epoch=1,
            amp="auto",
            grad_clip_norm=1.0,
            learning_rate=args.learning_rate,
            rss_cap_bytes=args.rss_cap_bytes,
            storage_cap_bytes=args.storage_cap_bytes,
        ),
        optimizer=torch.optim.SGD(model.parameters(), lr=args.learning_rate),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.jsonl"
    checkpoint_path = args.output_dir / "synthetic_paired_batch.pt"
    report_path = args.output_dir / "run_report.json"
    synthetic_config = SyntheticConfig(
        seed=23,
        visits=1,
        local_steps=1,
        raster_height=720,
        raster_width=1280,
        wavelength_nm=(450.0, 650.0, 1000.0),
    )

    started = time.time()
    reports = []
    with progress_path.open("w", encoding="utf-8") as handle:
        for pair_number, row in enumerate(rows, start=1):
            sample_index = int(row["sample_index"])
            batch = next(
                iter_paired_synthetic_training_batches(
                    synthetic_config,
                    sample_count=1,
                    start_index=sample_index,
                    device="cpu",
                )
            )
            report = trainer.train_epoch([batch], loss_fn=default_loss_fn)
            reports.append(report)
            record = {
                "pair_number": pair_number,
                "sample_index": sample_index,
                "views_trained": pair_number * 2,
                "loss": report.last_loss,
                "loss_is_finite": report.loss_is_finite,
                "rss_within_cap": report.rss_within_cap,
                "process_rss_bytes": report.process_rss_bytes,
                "peak_gpu_memory_bytes": report.peak_gpu_memory_bytes,
                "resource_cap_violations": list(report.resource_cap_violations),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            if pair_number % args.progress_every == 0 or pair_number == args.pair_count:
                handle.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
            if report.resource_cap_violations:
                raise RuntimeError(
                    f"resource cap violation at pair {pair_number}: "
                    f"{report.resource_cap_violations}"
                )

    torch.save(
        {
            "model": trainer.model.state_dict(),
            "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters()),
            "training_state": dataclasses.asdict(trainer.state),
            "training_config": {
                "batch_views": 2,
                "pair_count": args.pair_count,
                "start_index": args.start_index,
                "learning_rate": args.learning_rate,
                "rss_cap_bytes": args.rss_cap_bytes,
                "storage_cap_bytes": args.storage_cap_bytes,
            },
        },
        checkpoint_path,
    )
    max_rss = max((report.process_rss_bytes or 0) for report in reports)
    max_gpu = max((report.peak_gpu_memory_bytes or 0) for report in reports)
    report = {
        "status": "complete",
        "checkpoint_input": str(args.checkpoint),
        "checkpoint_input_sha256": _sha256(args.checkpoint),
        "checkpoint_output": str(checkpoint_path),
        "checkpoint_output_bytes": checkpoint_path.stat().st_size,
        "generation_dir": str(args.generation_dir),
        "pair_count": args.pair_count,
        "views_trained": args.pair_count * 2,
        "sample_index_first": args.start_index,
        "sample_index_last": args.start_index + args.pair_count - 1,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters()),
        "learning_rate": args.learning_rate,
        "rss_cap_bytes": args.rss_cap_bytes,
        "max_process_rss_bytes": max_rss,
        "max_peak_gpu_memory_bytes": max_gpu,
        "all_losses_finite": all(report.loss_is_finite for report in reports),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


def _validate_generation_manifest(
    generation_dir: Path, *, start_index: int, pair_count: int
) -> list[dict[str, object]]:
    paths = sorted(generation_dir.glob("shard-*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no generation shards found in {generation_dir}")
    rows: list[dict[str, object]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"generation row is not an object: {path}")
            rows.append(row)
    rows.sort(key=lambda row: int(row["sample_index"]))
    expected = list(range(start_index, start_index + pair_count))
    actual = [int(row["sample_index"]) for row in rows]
    if actual != expected:
        raise ValueError(
            "generation manifest indices do not exactly cover the training range: "
            f"expected {start_index}..{expected[-1]}, got {actual[:3]}..{actual[-3:]} "
            f"({len(actual)} rows)"
        )
    if any(row.get("targets") != [0.0, 1.0] for row in rows):
        raise ValueError("generation manifest contains a non-counterfactual pair")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
