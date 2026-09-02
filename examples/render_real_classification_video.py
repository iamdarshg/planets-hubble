"""Render real Kepler test-set classifications as a human-viewable MP4.

The classifier still consumes the compact 32x32 padded TPF representation.  The
video is a 1280x720 display artifact: it shows the six-channel diagnostic RGB
render, the aperture light curve, the ground-truth/model decision, and the
model probability for each test sequence.  It does not add detector
information to the model input.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SRC = ROOT / "src"
for path in (EXAMPLES, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from render_synthetic_transit_video import _visible_rgb  # noqa: E402
from train_kepler_real import _batch_records, _load_example, _load_model, _make_batch, _records  # noqa: E402


DISPLAY_SIZE = (1280, 720)


def _diagnostic_raster(example: dict[str, object]) -> np.ndarray:
    """Convert the trainer's normalized six-channel raster to display units."""

    raster = np.asarray(example["raster"], dtype=np.float32).copy()
    # The trainer's channel 0 is science / baseline - 1; the shared display
    # transform expects the corresponding positive physical ratio.
    raster[:, 0] = np.clip(1.0 + raster[:, 0], 1e-6, None)
    raster[:, 2] = np.clip(raster[:, 2], 0.0, None)
    raster[:, 3:] = np.clip(raster[:, 3:], 0.0, 1.0)
    return raster


def _lightcurve(example: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    raster = np.asarray(example["raster"], dtype=np.float32)
    valid = raster[:, 3] > 0.5
    values = np.where(valid, raster[:, 0], np.nan)
    curve = np.nanmean(values, axis=(1, 2))
    curve = np.nan_to_num(curve, nan=0.0).astype(np.float32) + 1.0
    times = np.asarray(example["times"], dtype=np.float64)
    return times - float(np.nanmedian(times)), curve


def _predict(
    model: torch.nn.Module,
    root: Path,
    records: list[dict[str, object]],
    batch_size: int,
) -> list[float]:
    probabilities: list[float] = []
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch_records in _batch_records(records, batch_size, seed=0, shuffle=False):
            batch = _make_batch(root, batch_records, device)
            output = model(batch)
            values = output["global_event_logits"].float().sigmoid().reshape(-1).cpu().tolist()
            probabilities.extend(float(value) for value in values)
            del batch, output
    return probabilities


def _render(
    records: list[dict[str, object]],
    probabilities: list[float],
    examples: list[dict[str, object]],
    output: Path,
    fps: int,
    physical_limits: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    rgb_frames: list[np.ndarray] = []
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    selected_frames: list[int] = []
    for example in examples:
        raster = _diagnostic_raster(example)
        rgb = _visible_rgb(raster, physical_limits)
        # Select the deepest valid photometric frame as the image panel.
        _, curve = _lightcurve(example)
        valid_count = raster[:, 3].sum(axis=(1, 2))
        score = np.where(valid_count > 0, curve, np.inf)
        selected = int(np.argmin(score))
        selected_frames.append(selected)
        rgb_frames.append(rgb[selected])
        curves.append(_lightcurve(example))

    figure, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    figure.patch.set_facecolor("#10131a")
    for axis in axes:
        axis.set_facecolor("#10131a")
        axis.tick_params(colors="white")
        for spine in axis.spines.values():
            spine.set_color("#667085")
    image = axes[0].imshow(rgb_frames[0], interpolation="nearest", aspect="auto", animated=True)
    axes[0].set_title("Real Kepler TPF | shared six-channel RGB diagnostic", color="white")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    line, = axes[1].plot([], [], color="#63b3ed", linewidth=2.0, label="aperture relative flux")
    marker, = axes[1].plot([], [], "o", color="#f6ad55", markersize=7, label="displayed frame")
    axes[1].set_title("Observed sequence light curve", color="white")
    axes[1].set_xlabel("Time relative to sequence median (days)", color="white")
    axes[1].set_ylabel("Relative aperture flux", color="white")
    axes[1].legend(facecolor="#10131a", labelcolor="white", loc="best")
    title = figure.suptitle("", color="white", fontsize=15)

    def update(index: int):
        record = records[index]
        example = examples[index]
        times, curve = curves[index]
        image.set_data(rgb_frames[index])
        line.set_data(times, curve)
        selected = selected_frames[index]
        marker.set_data([times[selected]], [curve[selected]])
        if times.size:
            pad = max(float(np.ptp(times)) * 0.05, 1e-5)
            axes[1].set_xlim(float(times.min() - pad), float(times.max() + pad))
        low = float(np.min(curve)) if curve.size else 0.99
        high = float(np.max(curve)) if curve.size else 1.01
        span = max(high - low, 1e-4)
        axes[1].set_ylim(low - span * 0.2, high + span * 0.2)
        label = int(record["label"])
        prediction = int(probabilities[index] >= 0.5)
        status = "CORRECT" if label == prediction else "ERROR"
        color = "#68d391" if status == "CORRECT" else "#fc8181"
        title.set_text(
            f"Test {index + 1}/{len(records)} | {record.get('target_name', 'unknown')} | "
            f"KepID {record.get('kepid', '?')} | GT={'TRANSIT' if label else 'CONTROL'} | "
            f"MODEL={'TRANSIT' if prediction else 'CONTROL'} p={probabilities[index]:.3f} | {status}"
        )
        title.set_color(color)
        return image, line, marker, title

    animation = FuncAnimation(figure, update, frames=len(records), interval=1000 / fps, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    actual_output = output
    if shutil.which("ffmpeg"):
        animation.save(actual_output, writer=FFMpegWriter(fps=fps, bitrate=2200), dpi=80)
    else:
        actual_output = output.with_suffix(".gif")
        animation.save(actual_output, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(figure)
    print(json.dumps({"output": str(actual_output), "frames": len(records), "fps": fps, "display_resolution": list(DISPLAY_SIZE)}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/synthetic-visuals/real_kepler_test_classification.mp4")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=208)
    args = parser.parse_args()
    if args.fps < 1 or args.batch_size < 1 or args.limit < 1:
        raise ValueError("fps, batch-size, and limit must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    root = args.manifest.parent
    records = [record for record in _records(args.manifest) if record.get("split") == "test"][: args.limit]
    if not records:
        raise ValueError("manifest has no test records")
    model = _load_model(args.checkpoint, torch.device("cpu"))
    probabilities = _predict(model, root, records, args.batch_size)
    examples = [_load_example(root, record) for record in records]
    physical_values = np.concatenate([
        _diagnostic_raster(example)[:, 0].reshape(-1)
        for example in examples
    ])
    physical_limits = (
        float(np.percentile(physical_values, 1.0)),
        float(np.percentile(physical_values, 99.5)),
    )
    _render(records, probabilities, examples, args.output, args.fps, physical_limits)

    labels = [int(record["label"]) for record in records]
    predictions = [int(value >= 0.5) for value in probabilities]
    correct = sum(label == prediction for label, prediction in zip(labels, predictions))
    metadata = {
        "status": "complete",
        "output": str(args.output),
        "source_manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "split": "test",
        "frames": len(records),
        "fps": args.fps,
        "display_resolution": list(DISPLAY_SIZE),
        "model_input_raster": [6, 32, 32],
        "model_input_note": "Kepler native 8x8 TPF values padded into 32x32; video is 1280x720 display only",
        "classification": {
            "samples": len(labels),
            "correct": correct,
            "accuracy": correct / len(labels),
            "true_positive": sum(label == prediction == 1 for label, prediction in zip(labels, predictions)),
            "true_negative": sum(label == prediction == 0 for label, prediction in zip(labels, predictions)),
            "false_positive": sum(prediction == 1 and label == 0 for label, prediction in zip(labels, predictions)),
            "false_negative": sum(prediction == 0 and label == 1 for label, prediction in zip(labels, predictions)),
            "mean_probability": float(np.mean(probabilities)),
        },
        "physical_display_limits": list(physical_limits),
        "labels": "Kepler DR25 confirmed-planet ephemeris labels in the local corpus manifest",
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata["classification"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
