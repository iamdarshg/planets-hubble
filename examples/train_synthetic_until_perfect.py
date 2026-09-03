"""Fast compact synthetic curriculum with an explicit zero-error holdout gate."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import AstroMambaHConfig  # noqa: E402
from synthetic import SyntheticConfig, SyntheticGenerator  # noqa: E402
from training.adapters import AstroMambaHInputs, AstroMambaHTrainingAdapter, AstroMambaHTrainingBatch  # noqa: E402
from training.harness import event_only_loss_fn, source_event_loss_fn  # noqa: E402


RSS_CAP_BYTES = 1_200_000_000  # strict 1.2 GB host RSS ceiling
CACHE_FORMAT_VERSION = 10


def _robust_temporal_score(values: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """Match the real adapter's invariant cadence-level flux-dip feature."""

    values = np.asarray(values, dtype=np.float32)
    uncertainty = np.asarray(uncertainty, dtype=np.float32)
    center = np.median(values, axis=(-1, -2), keepdims=True)
    mad_scale = np.median(np.abs(values - center), axis=(-1, -2), keepdims=True) * 1.4826
    scale = np.maximum(mad_scale, np.median(np.abs(uncertainty), axis=(-1, -2), keepdims=True))
    scale = np.maximum(scale, 1.0e-5)
    return np.clip((values - center) / scale, -20.0, 20.0).astype(np.float32)


def _config():
    return AstroMambaHConfig(
        input_height=32,
        input_width=32,
        allow_compact_input=True,
        source_top_k=1,
        context_token_count=4,
        stage_channels=(16, 32, 64, 96),
        embedding_dim=64,
        temporal_width=96,
        fusion_blocks=1,
        fusion_heads=4,
        temporal_blocks=2,
        canonical_wavelength_bins=8,
        wavelength_fourier_features=2,
        heatmap_rank=2,
        period_bin_count=8,
        period_feature_dim=4,
        decoder_width=64,
        decoder_blocks=1,
        spatial_chunk_size=1,
        decode_heatmaps=False,
    )


def _synthetic_config(seed: int) -> SyntheticConfig:
    return SyntheticConfig(
        seed=seed,
        visits=1,
        # Generate a compact detector cutout, then place it on the model's
        # required 32x32 canvas. This matches the real TPF adapter instead of
        # training on an unrealistically full-frame 32x32 scene.
        local_steps=16,
        # Retain the proven compact representation for this transition round;
        # the real-data footprint is introduced through masks and coverage
        # while the checkpoint adapts to the new nuisance statistics.
        raster_height=8,
        raster_width=8,
        wavelength_nm=(650.0,),
        wavelength_bandwidth_nm=80.0,
        exposure_seconds=1765.0,
        source_contrast=8.0,
        # Use count-domain rates that produce the small per-pixel uncertainty
        # of real Kepler apertures after normalization.  The previous
        # low-count settings made uncertainty itself a synthetic-only label
        # shortcut.
        source_rate_per_second=5000.0,
        background_rate_per_second=1000.0,
        pixel_noise_sigma=0.0002,
        # Kepler TPFs are compact crowded fields rather than isolated
        # single-source cutouts. Keep the supervised target at index zero,
        # but expose several unresolved neighbours whose planet-host status
        # is sampled independently and is not used as the event label.
        field_star_count=8,
        field_planet_probability=0.25,
        field_star_flux_ratio_min=0.03,
        field_star_flux_ratio_max=0.30,
        field_star_min_separation_pixels=1.5,
        # Kepler light curves contain percent-level correlated variability;
        # this is deliberately applied to null and injected counterfactuals
        # alike so the event label cannot be inferred from nuisance strength.
        variability_sigma=0.03,
        stellar_brightness_noise_sigma=0.003,
        stellar_brightness_ar1=0.85,
        stellar_brightness_amplitude_scatter=0.55,
        local_step_spacing_days=0.0204,
        timestamp_jitter_days=0.0005,
        interpolation_fraction=0.10,
        pointing_jitter_pixels=0.12,
        drift_pixels_per_visit=0.08,
        kepler_pointing_amplitude=0.001,
        kepler_thermal_amplitude=0.001,
        cosmic_ray_rate=0.005,
        radiation_hot_pixel_rate=0.002,
        cbv_common_mode_amplitude=0.001,
        barycentric_tdb_offset_seconds=120.0,
        light_time_correction_seconds=480.0,
        apparent_position_shift_arcsec=0.03,
        stellar_radial_velocity_mps=20_000.0,
        barycentric_radial_velocity_mps=29_000.0,
        gravitational_redshift_mps=636.0,
        orbital_radial_velocity_amplitude_mps=150.0,
    )


def _source_position(generator_config: SyntheticConfig, sample_index: int) -> tuple[float, float]:
    """Choose a deterministic, non-central source location for one sample."""

    rng = np.random.default_rng(generator_config.seed * 1_000_003 + sample_index)
    # Empirical target centroids in real train/validation TPFs cluster around
    # x=.60, y=.49 in their native cutouts, while retaining a non-central
    # spread. Keep the draw bounded without a deterministic center case.
    source_x, source_y = (
        float(value)
        for value in np.clip(rng.normal((0.60, 0.49), (0.09, 0.10)), 0.12, 0.88)
    )
    return source_x, source_y


def _embed_compact_view(view: dict[str, np.ndarray], *, canvas: int = 32) -> dict[str, np.ndarray]:
    """Place a compact detector cutout on the model's fixed-size canvas."""

    raster = view["raster"]
    if raster.ndim != 6:
        raise ValueError(f"expected batched raster, got {raster.shape}")
    _, visits, steps, channels, height, width = raster.shape
    if height > canvas or width > canvas:
        raise ValueError(f"compact detector {height}x{width} does not fit {canvas}x{canvas}")
    if (height, width) == (canvas, canvas):
        return view
    y0 = (canvas - height) // 2
    x0 = (canvas - width) // 2
    embedded = dict(view)
    raster_canvas = np.zeros((1, visits, steps, channels, canvas, canvas), dtype=np.float32)
    raster_canvas[:, :, :, :, y0 : y0 + height, x0 : x0 + width] = raster
    embedded["raster"] = raster_canvas
    objects = np.asarray(view["object_tokens"], dtype=np.float32).copy()
    objects[..., 0] = (x0 + objects[..., 0] * max(width - 1, 1)) / max(canvas - 1, 1)
    objects[..., 1] = (y0 + objects[..., 1] * max(height - 1, 1)) / max(canvas - 1, 1)
    objects[..., 2] *= max(width - 1, 1) / max(canvas - 1, 1)
    objects[..., 3] *= max(height - 1, 1) / max(canvas - 1, 1)
    embedded["object_tokens"] = objects
    return embedded


def _embed_source_xy(source_xy: np.ndarray, *, detector: int = 8, canvas: int = 32) -> np.ndarray:
    xy = np.asarray(source_xy, dtype=np.float32).copy()
    offset = (canvas - detector) // 2
    xy[:, 0] = (offset + xy[:, 0] * max(detector - 1, 1)) / max(canvas - 1, 1)
    xy[:, 1] = (offset + xy[:, 1] * max(detector - 1, 1)) / max(canvas - 1, 1)
    return xy


def _generate_pair(
    generator_config: SyntheticConfig,
    sample_index: int,
) -> tuple[list[dict[str, np.ndarray]], list[float], list[tuple[float, float]]]:
    """Generate one null/injected pair with an isolated deterministic seed."""

    source_x, source_y = _source_position(generator_config, sample_index)
    bundle_config = replace(
        generator_config,
        seed=generator_config.seed + sample_index,
        source_x=source_x,
        source_y=source_y,
    )
    bundle = SyntheticGenerator(bundle_config).generate()
    views: list[dict[str, np.ndarray]] = []
    labels: list[float] = []
    source_positions: list[tuple[float, float]] = []
    for view, label in ((bundle.null, 0.0), (bundle.injected, 1.0)):
        embedded = _embed_compact_view(
            bundle.as_model_numpy("null" if label == 0.0 else "injected")
        )
        # The real Kepler adapter exposes channel 4 as a good-quality mask,
        # whereas the generator stores an interpolation mask. Convert the
        # latter to the former at this adapter boundary.
        raster = embedded["raster"]
        wavelength_tokens = embedded["wavelength_tokens"]
        cadence_values = wavelength_tokens[..., 1]
        cadence_uncertainty = wavelength_tokens[..., 3]
        wavelength_tokens[..., 6] = _robust_temporal_score(
            cadence_values,
            cadence_uncertainty,
        )
        embedded["wavelength_tokens"] = wavelength_tokens
        raster[:, :, :, 4] = raster[:, :, :, 3] * (1.0 - raster[:, :, :, 4])
        embedded["raster"] = raster
        views.append(embedded)
        labels.append(label)
        source_positions.append((source_x, source_y))
    return views, labels, source_positions


def _make_batch(
    generator_config: SyntheticConfig,
    start_index: int,
    pair_count: int,
    device: torch.device,
    cache_path: Path | None = None,
) -> AstroMambaHTrainingBatch:
    cached_arrays: dict[str, np.ndarray] | None = None
    cached_labels: np.ndarray | None = None
    cached_source_xy: np.ndarray | None = None
    generated = False
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_arrays = {
                name: np.asarray(cached[name])
                for name in cached.files
                if name not in {"labels", "source_xy", "cache_version"}
            }
            cached_labels = np.asarray(cached["labels"], dtype=np.float32)
            cached_source_xy = np.asarray(cached["source_xy"], dtype=np.float32)
            cached_version = int(np.asarray(cached.get("cache_version", 0)).item())
        expected_samples = pair_count * 2
        if (
            cached_version != CACHE_FORMAT_VERSION
            or cached_labels.shape != (expected_samples,)
            or cached_source_xy.shape != (expected_samples, 2)
            or not cached_arrays
            or any(array.shape[0] != expected_samples for array in cached_arrays.values())
        ):
            cached_arrays = cached_labels = cached_source_xy = None

    if cached_arrays is None or cached_labels is None or cached_source_xy is None:
        generated = True
        views: list[dict[str, np.ndarray]] = []
        labels: list[float] = []
        source_positions: list[tuple[float, float]] = []
        pair_indices = range(start_index, start_index + pair_count)
        # Pair generation is independent by construction. Four bounded
        # workers improve CPU throughput without retaining a second full
        # dataset or changing the deterministic output order.
        worker_count = min(4, pair_count)
        if worker_count == 1:
            generated_pairs = map(
                lambda index: _generate_pair(generator_config, index),
                pair_indices,
            )
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            generated_pairs = executor.map(
                lambda index: _generate_pair(generator_config, index),
                pair_indices,
            )
        try:
            for pair_views, pair_labels, pair_positions in generated_pairs:
                views.extend(pair_views)
                labels.extend(pair_labels)
                source_positions.extend(pair_positions)
        finally:
            if worker_count > 1:
                executor.shutdown(wait=True)
        cached_arrays = {
            name: np.concatenate([view[name] for view in views], axis=0)
            for name in views[0]
        }
        cached_labels = np.asarray(labels, dtype=np.float32)
        cached_source_xy = np.asarray(source_positions, dtype=np.float32)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with temporary.open("wb") as stream:
                np.savez(
                    stream,
                    **cached_arrays,
                    labels=cached_labels,
                    source_xy=_embed_source_xy(cached_source_xy),
                    cache_version=np.asarray(CACHE_FORMAT_VERSION, dtype=np.int64),
                )
            os.replace(temporary, cache_path)

    if cached_source_xy is None:
        raise RuntimeError("source positions were not loaded")
    # New caches store canvas coordinates; freshly generated arrays still have
    # detector-local coordinates and are converted before tensor creation.
    if generated:
        cached_source_xy = _embed_source_xy(cached_source_xy)

    arrays = {name: torch.from_numpy(value) for name, value in cached_arrays.items()}
    # Keep trainable parameters and input tensors FP32. CUDA autocast is used
    # around the forward/backward compute, never for master weights.
    floating = torch.float32
    inputs = AstroMambaHInputs(
        **{
            name: value.to(device=device, dtype=floating if value.is_floating_point() else value.dtype)
            for name, value in arrays.items()
        },
        source_xy=torch.tensor(
            cached_source_xy,
            dtype=torch.float32,
            device=device,
        ),
    )
    return AstroMambaHTrainingBatch(
        inputs=inputs,
        target=torch.tensor(cached_labels[:, None], dtype=torch.float32, device=device),
        auxiliary_targets={
            "source_event": torch.tensor(cached_labels[:, None], dtype=torch.float32, device=device)
        },
    )


def _load_model(checkpoint: Path, device: torch.device) -> AstroMambaHTrainingAdapter:
    model = AstroMambaHTrainingAdapter(_config()).to(device, dtype=torch.float32)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model_state = model.state_dict()
    compatible = {
        name: value
        for name, value in state_dict.items()
        if name in model_state and tuple(value.shape) == tuple(model_state[name].shape)
    }
    result = model.load_state_dict(compatible, strict=False)
    model._checkpoint_transfer = {
        "checkpoint": str(checkpoint),
        "compatible_tensors": len(compatible),
        "model_tensors": len(model_state),
        "missing_after_transfer": len(result.missing_keys),
        "skipped_incompatible_tensors": len(state_dict) - len(compatible),
    }
    return model


def _new_model(device: torch.device) -> AstroMambaHTrainingAdapter:
    """Create FP32 master weights for a clean post-contract curriculum."""

    model = AstroMambaHTrainingAdapter(_config()).to(device, dtype=torch.float32)
    model._checkpoint_transfer = {"mode": "from_scratch", "compatible_tensors": 0}
    return model


def _rss() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return 0


def _batch_cache_path(cache_dir: Path, start_index: int, pair_count: int) -> Path:
    return cache_dir / f"batch_{start_index}_{pair_count}.npz"


def _evaluate(
    model,
    generator_config: SyntheticConfig,
    start_index: int,
    pair_count: int,
    batch_pairs: int,
    device: torch.device,
    cache_dir: Path,
) -> dict[str, object]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.inference_mode():
        for offset in range(0, pair_count, batch_pairs):
            count = min(batch_pairs, pair_count - offset)
            batch = _make_batch(
                generator_config,
                start_index + offset,
                count,
                device,
                _batch_cache_path(cache_dir, start_index + offset, count),
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
            probabilities.extend(float(value) for value in output["global_event_logits"].float().sigmoid().cpu())
            labels.extend([int(value) for value in batch.target.reshape(-1).cpu()])
            del batch, output
    predicted = [int(value >= 0.5) for value in probabilities]
    correct = sum(prediction == label for prediction, label in zip(predicted, labels))
    return {
        "pairs": pair_count,
        "samples": len(labels),
        "correct": correct,
        "errors": len(labels) - correct,
        "accuracy": correct / len(labels) if labels else None,
        "mean_probability": sum(probabilities) / len(probabilities) if probabilities else None,
        "true_positive": sum(prediction == label == 1 for prediction, label in zip(predicted, labels)),
        "true_negative": sum(prediction == label == 0 for prediction, label in zip(predicted, labels)),
        "false_positive": sum(prediction == 1 and label == 0 for prediction, label in zip(predicted, labels)),
        "false_negative": sum(prediction == 0 and label == 1 for prediction, label in zip(predicted, labels)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-checkpoint", type=Path)
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="rebuild the model with the corrected input contract instead of inheriting old weights",
    )
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=8192)
    parser.add_argument("--start-index", type=int, default=32768)
    parser.add_argument("--holdout-pairs", type=int, default=1024)
    parser.add_argument("--holdout-start-index", type=int, default=100000)
    parser.add_argument("--batch-pairs", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--loss-mode",
        choices=("event", "source"),
        default="event",
        help="event-only detector curriculum, or proposal-aware source auxiliary loss",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=8192)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--stop-holdout-errors",
        type=int,
        help="finish successfully after an epoch at or below this holdout error count",
    )
    parser.add_argument(
        "--rss-cap-bytes",
        type=int,
        default=RSS_CAP_BYTES,
        help="hard host RSS ceiling for this run; defaults to 1.4 GiB",
    )
    args = parser.parse_args()
    if min(args.pair_count, args.holdout_pairs, args.batch_pairs, args.max_epochs) < 1:
        raise ValueError("pair counts, batch-pairs, and max-epochs must be positive")
    if args.rss_cap_bytes < 1:
        raise ValueError("rss-cap-bytes must be positive")
    if args.stop_holdout_errors is not None and args.stop_holdout_errors < 0:
        raise ValueError("stop-holdout-errors must be non-negative")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cuda":
        # Benchmark search can reserve multi-gigabyte host workspaces on this
        # model, violating the 1.4 GiB RSS budget before the first update.
        # Keep algorithm selection bounded and use TF32 for the matmuls.
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if not args.from_scratch and (args.input_checkpoint is None or not args.input_checkpoint.is_file()):
        raise FileNotFoundError(args.input_checkpoint)
    model = _new_model(device) if args.from_scratch else _load_model(args.input_checkpoint, device)
    loss_fn = event_only_loss_fn if args.loss_mode == "event" else source_event_loss_fn
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator_config = _synthetic_config(seed=23)
    cache_dir = args.cache_dir or args.output_checkpoint.parent / "multistar-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    peak_rss = _rss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, object]] = []
    started = time.time()
    global_step = 0
    final_training = None
    final_holdout = None
    for epoch in range(args.max_epochs):
        model.train()
        losses: list[float] = []
        for offset in range(0, args.pair_count, args.batch_pairs):
            count = min(args.batch_pairs, args.pair_count - offset)
            batch = _make_batch(
                generator_config,
                args.start_index + offset,
                count,
                device,
                _batch_cache_path(cache_dir, args.start_index + offset, count),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
                loss = loss_fn(output, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite synthetic loss at step {global_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().float().cpu()))
            global_step += 1
            peak_rss = max(peak_rss, _rss())
            del batch, output, loss
            if peak_rss > args.rss_cap_bytes:
                raise RuntimeError(
                    f"RSS cap exceeded: {peak_rss} > {args.rss_cap_bytes}"
                )
        final_holdout = _evaluate(
            model,
            generator_config,
            args.holdout_start_index,
            args.holdout_pairs,
            args.batch_pairs,
            device,
            cache_dir,
        )
        final_training = _evaluate(
            model,
            generator_config,
            args.start_index,
            args.pair_count,
            args.batch_pairs,
            device,
            cache_dir,
        )
        record = {
            "epoch": epoch,
            "steps": len(losses),
            "mean_loss": sum(losses) / len(losses),
            "training": final_training,
            "holdout": final_holdout,
            "rss_bytes": peak_rss,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        epoch_checkpoint = args.output_checkpoint.with_suffix(f".epoch-{epoch}.pt")
        epoch_temporary = epoch_checkpoint.with_suffix(epoch_checkpoint.suffix + ".tmp")
        torch.save(
            {
                "model": model.state_dict(),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "checkpoint_transfer": getattr(model, "_checkpoint_transfer", {}),
                "optimizer_policy": "AdamW_FP32_master_weights",
                "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
                "amp_enabled": device.type == "cuda",
                "amp_dtype": "bfloat16" if device.type == "cuda" else None,
                "input_mode": "synthetic_detector_8x8_embedded_32x32_source_supervised",
                "loss_mode": args.loss_mode,
                "scene_model": "multi_star_field_target_counterfactual",
                "field_star_count": generator_config.field_star_count,
                "field_planet_probability": generator_config.field_planet_probability,
                "source_position_policy": "deterministic_noncentral_normal_center_0.60_0.49_spread_0.09_0.10",
                "cache_dir": str(cache_dir),
                "cache_enabled": True,
                "pair_count": args.pair_count,
                "sample_index_first": args.start_index,
                "sample_index_last": args.start_index + args.pair_count - 1,
                "holdout": final_holdout,
                "training": final_training,
                "history": history,
                "completed_epoch": epoch,
            },
            epoch_temporary,
        )
        os.replace(epoch_temporary, epoch_checkpoint)
        if final_training["errors"] == 0 or (
            args.stop_holdout_errors is not None
            and final_holdout["errors"] <= args.stop_holdout_errors
        ):
            break
    if final_holdout is None or final_training is None:
        raise RuntimeError("no synthetic epoch completed")
    temporary = args.output_checkpoint.with_suffix(args.output_checkpoint.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_transfer": getattr(model, "_checkpoint_transfer", {}),
            "optimizer_policy": "AdamW_FP32_master_weights",
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "amp_enabled": device.type == "cuda",
            "amp_dtype": "bfloat16" if device.type == "cuda" else None,
            "input_mode": "synthetic_detector_8x8_embedded_32x32_source_supervised",
            "loss_mode": args.loss_mode,
            "scene_model": "multi_star_field_target_counterfactual",
            "field_star_count": generator_config.field_star_count,
            "field_planet_probability": generator_config.field_planet_probability,
            "source_position_policy": "deterministic_noncentral_normal_center_0.60_0.49_spread_0.09_0.10",
            "cache_dir": str(cache_dir),
            "cache_enabled": True,
            "pair_count": args.pair_count,
            "sample_index_first": args.start_index,
            "sample_index_last": args.start_index + args.pair_count - 1,
            "holdout": final_holdout,
            "training": final_training,
            "history": history,
        },
        temporary,
    )
    os.replace(temporary, args.output_checkpoint)
    peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    report = {
        "status": (
            "overfit" if final_training["errors"] == 0 and final_holdout["errors"] > 0
            else "complete" if final_holdout["errors"] == 0
            else "training_not_overfit"
        ),
        "input_checkpoint": str(args.input_checkpoint) if args.input_checkpoint else None,
        "output_checkpoint": str(args.output_checkpoint),
        "checkpoint_transfer": getattr(model, "_checkpoint_transfer", {}),
        "pair_count": args.pair_count,
        "views_trained": args.pair_count * 2,
        "scene_model": "multi_star_field_target_counterfactual",
        "loss_mode": args.loss_mode,
        "field_star_count": generator_config.field_star_count,
        "field_planet_probability": generator_config.field_planet_probability,
        "source_position_policy": "deterministic_noncentral_normal_center_0.60_0.49_spread_0.09_0.10",
        "cache_dir": str(cache_dir),
        "cache_enabled": True,
        "sample_index_first": args.start_index,
        "sample_index_last": args.start_index + args.pair_count - 1,
        "holdout_start_index": args.holdout_start_index,
        "holdout": final_holdout,
        "training": final_training,
        "history": history,
        "device": str(device),
        "batch_pairs": args.batch_pairs,
        "max_epochs": args.max_epochs,
        "stop_holdout_errors": args.stop_holdout_errors,
        "global_steps": global_step,
        "rss_cap_bytes": args.rss_cap_bytes,
        "peak_process_rss_bytes": peak_rss,
        "rss_within_cap": peak_rss <= args.rss_cap_bytes,
        "peak_gpu_memory_allocated_bytes": peak_gpu,
        "elapsed_seconds": time.time() - started,
    }
    report_path = args.output_checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    gc.collect()
    return 0 if report["status"] == "complete" and report["rss_within_cap"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
