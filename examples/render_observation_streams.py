"""Render matched synthetic and real detector streams at true 1280x720 output.

The real Kepler target-pixel files are native 8x8 cutouts.  They are evaluated
on a 720x1280 grid using the maximum-degree tensor-product polynomial that the
native samples support; no claim of new detector information is made. Both
streams use the exact same six-channel-to-RGB diagnostic.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from render_synthetic_transit_video import (  # noqa: E402
    _config,
    _residual_rgb,
    _visible_rgb,
)
from synthetic import SyntheticGenerator  # noqa: E402


DISPLAY_SIZE = (1280, 720)


def _polynomial_resize(frames: np.ndarray, target_height: int, target_width: int, *, degree: int) -> np.ndarray:
    """Evaluate the highest-degree separable polynomial through each frame.

    For an 8x8 detector cutout, degree 7 in each axis is the highest
    interpolation degree that can be fit without inventing additional nodes.
    The native arrays remain available for all quantitative checks.  Masks are
    intentionally handled separately with nearest-neighbor semantics.
    """

    result = np.empty((frames.shape[0], target_height, target_width), dtype=np.float32)
    height, width = frames.shape[1:]
    if degree == 0:
        y_indices = np.rint(np.linspace(0, height - 1, target_height)).astype(np.int64)
        x_indices = np.rint(np.linspace(0, width - 1, target_width)).astype(np.int64)
        return frames[:, y_indices][:, :, x_indices].astype(np.float32, copy=False)
    degree_y = min(degree, height - 1)
    degree_x = min(degree, width - 1)
    nodes_y = np.arange(height, dtype=np.float64)
    nodes_x = np.arange(width, dtype=np.float64)
    targets_y = np.linspace(0.0, height - 1.0, target_height, dtype=np.float64)
    targets_x = np.linspace(0.0, width - 1.0, target_width, dtype=np.float64)

    def lagrange_weights(nodes: np.ndarray, targets: np.ndarray, fit_degree: int) -> np.ndarray:
        # Use a centered local stencil for numerical stability while retaining
        # the maximum available degree for this detector dimension.
        weights = np.zeros((targets.size, fit_degree + 1), dtype=np.float64)
        for target_index, target in enumerate(targets):
            center = int(np.clip(np.searchsorted(nodes, target), 0, len(nodes) - 1))
            first = int(np.clip(center - fit_degree // 2, 0, len(nodes) - fit_degree - 1))
            stencil = nodes[first : first + fit_degree + 1]
            for basis_index, node in enumerate(stencil):
                value = 1.0
                for other_index, other in enumerate(stencil):
                    if basis_index != other_index:
                        value *= (target - other) / (node - other)
                weights[target_index, basis_index] = value
        return weights

    weights_y = lagrange_weights(nodes_y, targets_y, degree_y)
    weights_x = lagrange_weights(nodes_x, targets_x, degree_x)
    for frame_index, frame in enumerate(frames):
        result[frame_index] = (
            weights_y @ np.asarray(frame, dtype=np.float64) @ weights_x.T
        ).astype(np.float32)
    return result


def _real_raster(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        science = np.asarray(data["science"], dtype=np.float32)
        uncertainty = np.asarray(data["uncertainty"], dtype=np.float32)
        finite = np.asarray(data["finite"], dtype=bool)
        quality = np.asarray(data["quality"], dtype=np.int32)
        time = np.asarray(data["time"], dtype=np.float64)
        label = int(np.asarray(data["label"]).item())
        kepid = int(np.asarray(data["source_kepid"]).item())
    valid = finite & np.isfinite(science) & np.isfinite(uncertainty)
    valid &= quality[:, None, None] == 0
    finite_values = science[valid]
    baseline = float(np.nanmedian(finite_values)) if finite_values.size else 1.0
    baseline = max(abs(baseline), 1e-6)
    physical = np.nan_to_num(science / baseline, nan=1.0, posinf=1.0, neginf=1.0)
    sigma = np.nan_to_num(np.abs(uncertainty) / baseline, nan=0.0, posinf=0.0, neginf=0.0)
    residual = (physical - 1.0) / np.maximum(sigma, np.nanmedian(sigma[valid]) if np.any(valid) else 1e-6)
    native_raster = np.stack(
        (
            physical,
            np.clip(residual, -50.0, 50.0),
            sigma,
            valid.astype(np.float32),
            np.zeros_like(physical),
            valid.astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    raster = np.empty((science.shape[0], 6, DISPLAY_SIZE[1], DISPLAY_SIZE[0]), dtype=np.float32)
    for channel in (0, 1, 2):
        raster[:, channel] = _polynomial_resize(
            native_raster[:, channel], DISPLAY_SIZE[1], DISPLAY_SIZE[0], degree=7
        )
    for channel in (3, 4, 5):
        raster[:, channel] = _polynomial_resize(
            native_raster[:, channel], DISPLAY_SIZE[1], DISPLAY_SIZE[0], degree=0
        )
    raster[:, 2] = np.clip(raster[:, 2], 0.0, None)
    raster[:, 3:] = np.clip(raster[:, 3:], 0.0, 1.0)
    return raster, {
        "path": str(path),
        "native_raster_shape": list(science.shape[1:]),
        "display_resolution": list(DISPLAY_SIZE),
        "resampling": "separable maximum-degree polynomial interpolation (degree 7 for 8x8 science channels); nearest for masks",
        "label": label,
        "kepid": kepid,
        "time": time.tolist(),
        "baseline_flux": baseline,
        "quality_nonzero_frames": int(np.count_nonzero(quality)),
    }


def _hst_raster(paths: list[Path]) -> tuple[np.ndarray, dict[str, object]]:
    """Load prepared HST/WFC3 images already normalized to 720x1280."""

    if not paths:
        raise ValueError("HST path list is empty")
    science_frames = []
    uncertainty_frames = []
    valid_frames = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            science = np.asarray(data["science"], dtype=np.float32)
            uncertainty = np.asarray(data["uncertainty"], dtype=np.float32)
            dq = np.asarray(data["dq"])
        if science.shape != DISPLAY_SIZE[::-1] or uncertainty.shape != science.shape:
            raise ValueError(f"prepared HST frame is not 720x1280: {path}")
        valid = np.isfinite(science) & np.isfinite(uncertainty) & (dq == 0)
        science_frames.append(science)
        uncertainty_frames.append(uncertainty)
        valid_frames.append(valid)
    science_stack = np.stack(science_frames)
    uncertainty_stack = np.stack(uncertainty_frames)
    valid_stack = np.stack(valid_frames)
    positive = science_stack[valid_stack & (science_stack > 0.0)]
    baseline = float(np.median(positive)) if positive.size else 1.0
    baseline = max(abs(baseline), 1e-6)
    physical = np.nan_to_num(science_stack / baseline, nan=1.0, posinf=1.0, neginf=1.0)
    sigma = np.nan_to_num(np.abs(uncertainty_stack) / baseline, nan=0.0, posinf=0.0, neginf=0.0)
    robust_sigma = max(float(np.nanmedian(sigma[valid_stack])), 1e-6) if np.any(valid_stack) else 1e-6
    residual = np.clip((physical - 1.0) / robust_sigma, -50.0, 50.0)
    raster = np.stack(
        (
            physical,
            residual,
            sigma,
            valid_stack.astype(np.float32),
            np.zeros_like(physical),
            valid_stack.astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    return raster, {
        "instrument": "HST/WFC3-IR",
        "paths": [str(path) for path in paths],
        "native_raster_shape": list(science_stack.shape[1:]),
        "display_resolution": list(DISPLAY_SIZE),
        "resampling": "none; prepared HST frame already has the 720x1280 model canvas",
        "baseline_flux": baseline,
    }


def _save_animation(frames: np.ndarray, output: Path, title: str, subtitle: str, fps: int, rgb_limits: tuple[float, float], *, residual: np.ndarray | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    output.parent.mkdir(parents=True, exist_ok=True)
    rgb = _visible_rgb(frames, rgb_limits)
    figure, axes = plt.subplots(1, 2 if residual is not None else 1, figsize=(16, 9), constrained_layout=True)
    axes = np.atleast_1d(axes)
    figure.suptitle(title)
    image = axes[0].imshow(rgb[0], interpolation="nearest", animated=True)
    axes[0].set_title(subtitle)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    images = [image]
    if residual is not None:
        residual_image = axes[1].imshow(_residual_rgb(residual[0]), interpolation="nearest", animated=True)
        axes[1].set_title("Positive - control signed residual")
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        images.append(residual_image)

    def update(frame: int):
        image.set_data(rgb[frame])
        if residual is not None:
            residual_image.set_data(_residual_rgb(residual[frame]))
        return tuple(images)

    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps, blit=False)
    if shutil.which("ffmpeg"):
        animation.save(output, writer=FFMpegWriter(fps=fps, bitrate=2200), dpi=80)
    else:
        fallback = output.with_suffix(".gif")
        animation.save(fallback, writer=PillowWriter(fps=fps), dpi=80)
        output = fallback
    plt.close(figure)
    print(json.dumps({"output": str(output), "frames": len(frames), "display_resolution": list(DISPLAY_SIZE)}), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, default=ROOT / "artifacts/real-transit-acquisition/kepler-1000/examples")
    parser.add_argument("--real-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/synthetic-visuals")
    parser.add_argument("--hst-dir", type=Path, default=ROOT / "data/real/hd209458_prepared")
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    control_path = args.real_dir / f"{args.real_index:04d}-control.npz"
    positive_path = args.real_dir / f"{args.real_index:04d}-positive.npz"
    if not control_path.is_file() or not positive_path.is_file():
        raise FileNotFoundError(f"missing real pair: {control_path} / {positive_path}")
    control, control_meta = _real_raster(control_path)
    positive, positive_meta = _real_raster(positive_path)
    real_limits = (
        float(np.percentile(np.concatenate((control[:, 0].ravel(), positive[:, 0].ravel())), 1.0)),
        float(np.percentile(np.concatenate((control[:, 0].ravel(), positive[:, 0].ravel())), 99.5)),
    )
    _save_animation(
        control,
        args.output_dir / f"real_{args.real_index:04d}_control.mp4",
        "Real Kepler control stream | 8x8 native detector cutout -> 1280x720 display",
        "real control, six-channel diagnostic RGB",
        args.fps,
        real_limits,
    )
    _save_animation(
        positive,
        args.output_dir / f"real_{args.real_index:04d}_positive.mp4",
        "Real Kepler positive stream | 8x8 native detector cutout -> 1280x720 display",
        "real confirmed-transit window, six-channel diagnostic RGB",
        args.fps,
        real_limits,
    )
    _save_animation(
        positive,
        args.output_dir / f"real_{args.real_index:04d}_pair_residual.mp4",
        "Real Kepler pair diagnostic | 1280x720 display",
        "real positive stream",
        args.fps,
        real_limits,
        residual=positive[:, 1] - control[:, 1],
    )

    synthetic_bundle = SyntheticGenerator(_config(20260831)).generate()
    synthetic = synthetic_bundle.injected.raster[0]
    synthetic_limits = (
        float(np.percentile(synthetic[:, 0].ravel(), 1.0)),
        float(np.percentile(synthetic[:, 0].ravel(), 99.5)),
    )
    _save_animation(
        synthetic,
        args.output_dir / "synthetic_10star_injected.mp4",
        "Synthetic multi-star transit stream | 32x32 native raster -> 1280x720 display",
        "synthetic injected target transit, six-channel diagnostic RGB",
        args.fps,
        synthetic_limits,
    )
    hst_paths = sorted(args.hst_dir.glob("*.npz"))
    if hst_paths:
        hst, hst_meta = _hst_raster(hst_paths)
        hst_limits = (
            float(np.percentile(hst[:, 0].ravel(), 1.0)),
            float(np.percentile(hst[:, 0].ravel(), 99.5)),
        )
        _save_animation(
            hst,
            args.output_dir / "hst_wfc3_hd209458.mp4",
            "Real HST/WFC3-IR stream | native prepared 720x1280 model canvas",
            "HD 209458 repeated WFC3/IR detector frames, six-channel diagnostic RGB",
            args.fps,
            hst_limits,
        )
    else:
        hst_meta = {"status": "not_found", "directory": str(args.hst_dir)}
    metadata = {
        "display_resolution": list(DISPLAY_SIZE),
        "rgb_mapping": "R=log physical flux, G=signed noise-scaled residual, B=inverse uncertainty; validity*coverage gate",
        "real_control": control_meta,
        "real_positive": positive_meta,
        "real_rgb_limits": real_limits,
        "synthetic_rgb_limits": synthetic_limits,
        "synthetic_native_raster_shape": list(synthetic.shape),
        "hst": hst_meta,
        "real_native_resolution_warning": "Real Kepler examples are native 8x8 target-pixel cutouts; degree-7 interpolation and 1280x720 encoding improve display/model shape only, not native spatial information.",
    }
    (args.output_dir / "observation_streams.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
