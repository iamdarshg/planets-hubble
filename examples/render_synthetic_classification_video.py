"""Render model classifications over a held-out synthetic test stream."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
for path in (SRC, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from render_observation_streams import _visible_rgb  # noqa: E402
from train_synthetic_until_perfect import (  # noqa: E402
    _batch_cache_path,
    _config,
    _make_batch,
    _synthetic_config,
    _load_model,
)


DISPLAY_SIZE = (1280, 720)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts/training/final-synthetic-continued/synthetic_continued.epoch-7.pt",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "artifacts/training/final-synthetic-continued/multistar-cache",
    )
    parser.add_argument("--test-start-index", type=int, default=400000)
    parser.add_argument("--test-pairs", type=int, default=128)
    parser.add_argument("--cache-pairs", type=int, default=512)
    parser.add_argument("--batch-pairs", type=int, default=64)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/synthetic-visuals/synthetic_test_classification.mp4",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.test_pairs < 1 or args.cache_pairs < args.test_pairs:
        raise ValueError("cache-pairs must cover test-pairs")

    device = torch.device("cpu")
    generator_config = _synthetic_config(seed=23)
    cache_path = _batch_cache_path(args.cache_dir, args.test_start_index, args.cache_pairs)
    batch = _make_batch(
        generator_config,
        args.test_start_index,
        args.cache_pairs,
        device,
        cache_path,
    )
    view_count = args.test_pairs * 2
    if view_count > batch.batch_size:
        raise ValueError("test-pairs exceeds cached batch")
    batch = replace(
        batch,
        inputs=replace(
            batch.inputs,
            raster=batch.inputs.raster[:view_count],
            wavelength_tokens=batch.inputs.wavelength_tokens[:view_count],
            wavelength_mask=batch.inputs.wavelength_mask[:view_count],
            object_tokens=batch.inputs.object_tokens[:view_count],
            object_mask=batch.inputs.object_mask[:view_count],
            geometry=batch.inputs.geometry[:view_count],
            exposure_duration=batch.inputs.exposure_duration[:view_count],
            coverage_vector=batch.inputs.coverage_vector[:view_count],
            local_time=batch.inputs.local_time[:view_count],
            long_time=batch.inputs.long_time[:view_count],
            source_xy=batch.inputs.source_xy[:view_count],
        ),
        target=batch.target[:view_count],
        auxiliary_targets={
            name: value[:view_count] for name, value in batch.auxiliary_targets.items()
        },
    )

    model = _load_model(args.checkpoint, device)
    model.eval()
    probabilities: list[float] = []
    with torch.inference_mode():
        for start in range(0, view_count, args.batch_pairs * 2):
            end = min(view_count, start + args.batch_pairs * 2)
            chunk = replace(
                batch,
                inputs=replace(
                    batch.inputs,
                    raster=batch.inputs.raster[start:end],
                    wavelength_tokens=batch.inputs.wavelength_tokens[start:end],
                    wavelength_mask=batch.inputs.wavelength_mask[start:end],
                    object_tokens=batch.inputs.object_tokens[start:end],
                    object_mask=batch.inputs.object_mask[start:end],
                    geometry=batch.inputs.geometry[start:end],
                    exposure_duration=batch.inputs.exposure_duration[start:end],
                    coverage_vector=batch.inputs.coverage_vector[start:end],
                    local_time=batch.inputs.local_time[start:end],
                    long_time=batch.inputs.long_time[start:end],
                    source_xy=batch.inputs.source_xy[start:end],
                ),
                target=batch.target[start:end],
                auxiliary_targets={
                    name: value[start:end] for name, value in batch.auxiliary_targets.items()
                },
            )
            output = model(chunk)
            probabilities.extend(
                float(value)
                for value in output["global_event_logits"].float().sigmoid().reshape(-1).cpu()
            )

    raster = batch.inputs.raster[:, 0, 0].numpy()
    labels = batch.target.reshape(-1).numpy().astype(np.int64)
    limits = (
        float(np.percentile(raster[:, 0], 1.0)),
        float(np.percentile(raster[:, 0], 99.5)),
    )
    rgb = _visible_rgb(raster, limits)
    predicted = np.asarray(probabilities) >= 0.5
    correct = predicted == labels

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    image = axis.imshow(rgb[0], interpolation="nearest", animated=True)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("black")

    def update(frame: int):
        label = int(labels[frame])
        probability = float(probabilities[frame])
        prediction = int(predicted[frame])
        status = "CORRECT" if correct[frame] else "ERROR"
        status_color = "lime" if correct[frame] else "red"
        image.set_data(rgb[frame])
        axis.set_title(
            f"Synthetic holdout classification | view {frame + 1}/{view_count} | "
            f"GT: {'TRANSIT' if label else 'CONTROL'} | "
            f"MODEL: {'TRANSIT' if prediction else 'CONTROL'} | "
            f"p(transit)={probability:.4f} | {status}",
            color=status_color,
            fontsize=15,
        )
        return (image,)

    animation = FuncAnimation(figure, update, frames=view_count, interval=1000 / args.fps, blit=False)
    if shutil.which("ffmpeg"):
        animation.save(args.output, writer=FFMpegWriter(fps=args.fps, bitrate=2600), dpi=80)
        actual_output = args.output
    else:
        actual_output = args.output.with_suffix(".gif")
        animation.save(actual_output, writer=PillowWriter(fps=args.fps), dpi=80)
    plt.close(figure)

    metadata = {
        "output": str(actual_output),
        "checkpoint": str(args.checkpoint),
        "test_start_index": args.test_start_index,
        "pairs": args.test_pairs,
        "views": view_count,
        "display_resolution": list(DISPLAY_SIZE),
        "native_model_raster": list(raster.shape[1:]),
        "correct": int(correct.sum()),
        "errors": int((~correct).sum()),
        "accuracy": float(correct.mean()),
        "true_positive": int(np.count_nonzero(predicted & (labels == 1))),
        "true_negative": int(np.count_nonzero(~predicted & (labels == 0))),
        "false_positive": int(np.count_nonzero(predicted & (labels == 0))),
        "false_negative": int(np.count_nonzero(~predicted & (labels == 1))),
        "probability_threshold": 0.5,
        "display_transform": "shared six-channel RGB compression; 32x32 model raster displayed at 1280x720",
    }
    metadata_path = actual_output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
