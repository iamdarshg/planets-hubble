"""Train on transits injected into real Kepler control windows.

This is a domain-randomized bridge between the analytic simulator and the
observed Kepler corpus. Only ``label=0`` records from the manifest's train
split are used as parents. The held-out real validation/test windows remain
untouched; the injected positive is created in memory and is never written as
if it were an observed detection.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from train_kepler_real import (  # noqa: E402
    _event_loss,
    _load_example,
    _load_model,
    _metrics,
    _records,
    _rss_bytes,
)
from training.adapters import AstroMambaHInputs, AstroMambaHTrainingBatch  # noqa: E402


RSS_CAP_BYTES = 840_000_000


def _robust_temporal_score(values: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    center = float(np.median(values))
    scale = max(float(np.median(np.abs(values - center))) * 1.4826, float(np.median(np.abs(uncertainty))), 1.0e-5)
    return np.clip((values - center) / scale, -20.0, 20.0).astype(np.float32)


def _copy_example(example: dict[str, object]) -> dict[str, object]:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in example.items()
    }


def _conditioned_pair(root: Path, record: dict[str, object], rng: np.random.Generator) -> tuple[dict[str, object], dict[str, object]]:
    null = _load_example(root, record)
    injected = _copy_example(null)
    raster = injected["raster"]
    tokens = injected["wavelength_tokens"]
    times = np.asarray(injected["times"], dtype=np.float64)
    objects = np.asarray(injected["object_tokens"], dtype=np.float32)
    if not isinstance(raster, np.ndarray) or not isinstance(tokens, np.ndarray) or not isinstance(objects, np.ndarray):
        raise TypeError("real adapter returned non-array example fields")

    # The source token gives the observed target location in the padded
    # 32x32 canvas. A compact Gaussian is an instrument-agnostic PSF prior,
    # while every background/noise/quality pattern remains from the real TPF.
    source_x = float(objects[0, 0]) * 32.0
    source_y = float(objects[0, 1]) * 32.0
    yy, xx = np.mgrid[0:32, 0:32]
    sigma = float(rng.uniform(0.8, 1.5))
    psf = np.exp(-0.5 * (((xx - source_x) / sigma) ** 2 + ((yy - source_y) / sigma) ** 2))
    psf /= max(float(psf.max()), 1.0e-8)
    cadence = float(np.median(np.diff(times))) if len(times) > 1 else 0.0204
    event_center = float(np.median(times) + rng.uniform(-cadence, cadence))
    duration_days = float(rng.uniform(0.035, 0.14))
    depth = float(rng.uniform(0.003, 0.025))
    profile = np.exp(-0.5 * ((times - event_center) / max(duration_days / 2.0, 1.0e-5)) ** 2)
    delta = depth * profile[:, None, None] * psf[None, :, :]
    valid = np.asarray(raster[:, 3], dtype=np.float32) > 0.0
    delta *= valid
    raster[:, 0] -= delta
    raster[:, 1] -= delta

    # The aperture token represents the target-star light curve rather than a
    # single pixel, so its fractional drop should be the transit depth itself.
    source_drop = depth * profile
    tokens[:, 0, 1] -= source_drop
    tokens[:, 0, 2] -= source_drop / np.maximum(tokens[:, 0, 3], 1.0e-3)
    tokens[:, 0, 6] = _robust_temporal_score(tokens[:, 0, 1], tokens[:, 0, 3])
    injected["raster"] = raster
    injected["wavelength_tokens"] = tokens
    injected["label"] = 1
    null["label"] = 0
    return null, injected


def _batch_from_examples(examples: list[dict[str, object]], device: torch.device) -> AstroMambaHTrainingBatch:
    batch_size = len(examples)
    frames = max(int(np.asarray(example["raster"]).shape[0]) for example in examples)
    raster = np.zeros((batch_size, 1, frames, 6, 32, 32), dtype=np.float32)
    wavelength = np.zeros((batch_size, 1, frames, 1, 8), dtype=np.float32)
    geometry = np.zeros((batch_size, 1, frames, 10), dtype=np.float32)
    coverage = np.zeros((batch_size, 1, frames, 6), dtype=np.float32)
    local_time = np.zeros((batch_size, 1, frames, 5), dtype=np.float32)
    long_time = np.zeros((batch_size, 1, 5), dtype=np.float32)
    objects = np.zeros((batch_size, 1, 1, 12), dtype=np.float32)
    step_mask = np.zeros((batch_size, 1, frames), dtype=bool)
    labels: list[float] = []
    for index, example in enumerate(examples):
        count = int(np.asarray(example["raster"]).shape[0])
        raster[index, 0, :count] = np.asarray(example["raster"])
        wavelength[index, 0, :count] = np.asarray(example["wavelength_tokens"])
        geometry[index, 0, :count] = np.asarray(example["geometry"])
        coverage[index, 0, :count] = np.asarray(example["coverage"])
        local_time[index, 0, :count] = np.asarray(example["local_time"])
        long_time[index, 0] = np.asarray(example["long_time"])
        objects[index, 0] = np.asarray(example["object_tokens"])
        step_mask[index, 0, :count] = True
        labels.append(float(example["label"]))
    floating = torch.float32
    inputs = AstroMambaHInputs(
        raster=torch.from_numpy(raster).to(device=device, dtype=floating),
        wavelength_tokens=torch.from_numpy(wavelength).to(device=device, dtype=floating),
        wavelength_mask=torch.ones((batch_size, 1, frames, 1), dtype=torch.bool, device=device),
        object_tokens=torch.from_numpy(objects).to(device=device, dtype=floating),
        object_mask=torch.ones((batch_size, 1, 1), dtype=torch.bool, device=device),
        geometry=torch.from_numpy(geometry).to(device=device, dtype=floating),
        exposure_duration=torch.full((batch_size, 1, frames, 1), 1800.0, dtype=floating, device=device),
        coverage_vector=torch.from_numpy(coverage).to(device=device, dtype=floating),
        local_time=torch.from_numpy(local_time).to(device=device, dtype=floating),
        long_time=torch.from_numpy(long_time).to(device=device, dtype=floating),
        visit_mask=torch.ones((batch_size, 1), dtype=torch.bool, device=device),
        step_mask=torch.from_numpy(step_mask).to(device=device),
        source_xy=torch.from_numpy(objects[:, 0, 0, :2]).to(device=device, dtype=floating),
    )
    target = torch.tensor(labels, dtype=floating, device=device).reshape(-1, 1)
    return AstroMambaHTrainingBatch(inputs=inputs, target=target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--pairs-per-batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rss-cap-bytes", type=int, default=RSS_CAP_BYTES)
    args = parser.parse_args()
    if min(args.pairs_per_batch, args.epochs) < 1 or args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("pairs-per-batch and epochs must be positive; learning-rate must be positive")
    if args.rss_cap_bytes < 1:
        raise ValueError("rss-cap-bytes must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    records = _records(args.manifest)
    train_controls = [record for record in records if record.get("split") == "train" and int(record["label"]) == 0]
    validation = [record for record in records if record.get("split") == "validation"]
    if not train_controls or not validation:
        raise ValueError("manifest needs train controls and validation records")
    model = _load_model(args.input_checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    peak_rss = _rss_bytes()
    history: list[dict[str, object]] = []
    best_bce = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        order = np.arange(len(train_controls))
        np.random.default_rng(args.seed + epoch).shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), args.pairs_per_batch):
            examples: list[dict[str, object]] = []
            for index in order[start : start + args.pairs_per_batch]:
                null, injected = _conditioned_pair(args.manifest.parent, train_controls[int(index)], rng)
                examples.extend((null, injected))
            batch = _batch_from_examples(examples, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = _event_loss(output, batch, 0.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            peak_rss = max(peak_rss, _rss_bytes())
            if peak_rss > args.rss_cap_bytes:
                raise MemoryError(f"RSS cap exceeded: {peak_rss} > {args.rss_cap_bytes}")
            del batch, output, loss
        with torch.inference_mode():
            validation_metrics = _metrics(model, args.manifest.parent, validation, device, args.pairs_per_batch)
        validation_bce = float(validation_metrics["mean_bce_loss"])
        history.append({"epoch": epoch, "steps": len(losses), "mean_loss": sum(losses) / len(losses), "validation": validation_metrics})
        if validation_bce < best_bce:
            best_bce = validation_bce
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    if best_state is None:
        raise RuntimeError("no conditioned-training checkpoint was produced")
    model.load_state_dict(best_state)
    test_metrics = _metrics(model, args.manifest.parent, [record for record in records if record.get("split") == "test"], device, args.pairs_per_batch)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_checkpoint.with_suffix(args.output_checkpoint.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "base_checkpoint": str(args.input_checkpoint), "training_history": history, "input_mode": "real_kepler_control_parent_in_memory_transit_injection", "best_epoch": best_epoch, "best_validation_bce": best_bce}, temporary)
    os.replace(temporary, args.output_checkpoint)
    report = {"status": "complete", "manifest": str(args.manifest), "input_checkpoint": str(args.input_checkpoint), "output_checkpoint": str(args.output_checkpoint), "device": str(device), "epochs": args.epochs, "pairs_per_batch": args.pairs_per_batch, "train_control_parents": len(train_controls), "validation": history[-1]["validation"], "test": test_metrics, "history": history, "best_epoch": best_epoch, "best_validation_bce": best_bce, "peak_process_rss_bytes": peak_rss, "rss_cap_bytes": args.rss_cap_bytes, "rss_within_cap": peak_rss <= args.rss_cap_bytes, "elapsed_seconds": time.time() - start_time}
    args.output_checkpoint.with_suffix(".report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "validation": report["validation"], "test": report["test"], "peak_process_rss_bytes": peak_rss, "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
