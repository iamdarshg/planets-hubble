"""Fast compact synthetic curriculum with an explicit zero-error holdout gate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import shutil
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
CACHE_FORMAT_VERSION = 17


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
        field_planet_probability=0.45,
        field_star_flux_ratio_min=0.03,
        field_star_flux_ratio_max=0.45,
        field_star_min_separation_pixels=1.5,
        # Kepler light curves contain percent-level correlated variability;
        # this is deliberately applied to null and injected counterfactuals
        # alike so the event label cannot be inferred from nuisance strength.
        # The real Kepler adapter's aperture residuals are typically a few
        # uncertainty units, not tens of units. Keep the target-star common
        # mode near the measured fractional fluctuation scale; the separate
        # field-star AR(1) process below still supplies crowded-field noise.
        variability_sigma=0.0015,
        stellar_brightness_noise_sigma=0.0015,
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


def _sample_target_event_config(
    generator_config: SyntheticConfig,
    sample_index: int,
    *,
    transit_radius_ratio_min: float = 0.006,
    transit_radius_ratio_max: float = 0.04,
) -> SyntheticConfig:
    """Vary target-transit morphology while keeping every positive observable.

    A fixed phase, duration, depth, and impact parameter lets a classifier
    memorize one artificial waveform. The real corpus contains a range of
    transit shapes, so sample those parameters deterministically per scene.
    The epoch stays inside the compact 16-cadence window, which preserves the
    counterfactual label contract instead of creating positive labels with no
    visible event.
    """

    rng = np.random.default_rng(generator_config.seed * 7_919 + sample_index * 104_729)
    window_days = generator_config.local_step_spacing_days * max(
        generator_config.local_steps - 1,
        1,
    )
    epoch_offset = float(rng.uniform(0.04, max(0.041, window_days - 0.035)))
    return replace(
        generator_config,
        transit_period_days=float(rng.uniform(2.0, 30.0)),
        transit_epoch_offset_days=epoch_offset,
        transit_duration_hours=float(rng.uniform(1.5, 6.0)),
        transit_radius_ratio=float(rng.uniform(transit_radius_ratio_min, transit_radius_ratio_max)),
        transit_impact_parameter=float(rng.uniform(0.0, 0.85)),
        invalid_exposures=_sample_invalid_exposures(generator_config, rng),
    )


def _apply_curriculum_overrides(
    generator_config: SyntheticConfig,
    *,
    field_star_count: int | None,
    field_planet_probability: float | None,
    stellar_brightness_noise_sigma: float | None,
) -> SyntheticConfig:
    """Apply explicit synthetic difficulty controls for staged curricula."""

    updates: dict[str, object] = {}
    if field_star_count is not None:
        updates["field_star_count"] = field_star_count
    if field_planet_probability is not None:
        updates["field_planet_probability"] = field_planet_probability
    if stellar_brightness_noise_sigma is not None:
        updates["stellar_brightness_noise_sigma"] = stellar_brightness_noise_sigma
    return replace(generator_config, **updates) if updates else generator_config


def _sample_invalid_exposures(
    generator_config: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    """Inject occasional sparse-cadence gaps to match real compact TPFs."""

    if rng.random() >= 0.25:
        return ()
    step = int(rng.integers(0, generator_config.local_steps))
    return ((0, step),)


def _match_real_adapter_context(
    embedded: dict[str, np.ndarray],
    *,
    timestamps_mid_bjd_tdb: np.ndarray,
    detector_height: int,
    detector_width: int,
    canvas: int = 32,
) -> dict[str, np.ndarray]:
    """Make simulator side channels obey the real Kepler adapter contract."""

    result = dict(embedded)
    mid = np.asarray(timestamps_mid_bjd_tdb, dtype=np.float64)
    finite_mid = mid[np.isfinite(mid)]
    center = float(np.median(finite_mid))
    cadence_days = (
        float(np.median(np.diff(mid[0][np.isfinite(mid[0])])))
        if mid.shape[1] > 1 and np.isfinite(mid[0]).sum() > 1
        else 0.0204
    )
    result["local_time"] = np.stack(
        (mid - center, mid - center, mid - center,
         np.full_like(mid, cadence_days), np.ones_like(mid)),
        axis=-1,
    ).astype(np.float32)[None, ...]
    result["long_time"] = np.asarray(
        [float(finite_mid.min() - center), float(finite_mid.max() - center),
         float(finite_mid.size), float(finite_mid.max() - finite_mid.min()), 1.0],
        dtype=np.float32,
    )[None, None, :]
    result["geometry"] = np.zeros_like(result["geometry"], dtype=np.float32)

    raster = result["raster"]
    y0 = (canvas - detector_height) // 2
    x0 = (canvas - detector_width) // 2
    compact_valid = raster[0, :, :, 3, y0 : y0 + detector_height, x0 : x0 + detector_width]
    compact_quality = raster[0, :, :, 4, y0 : y0 + detector_height, x0 : x0 + detector_width]
    valid_fraction = compact_valid.mean(axis=(-1, -2))
    quality_fraction = compact_quality.mean(axis=(-1, -2))
    coverage = np.stack(
        (np.ones_like(valid_fraction), valid_fraction, quality_fraction,
         np.full_like(valid_fraction, detector_height / canvas),
         np.full_like(valid_fraction, detector_width / canvas),
         np.ones_like(valid_fraction)),
        axis=-1,
    )
    result["coverage_vector"] = coverage[None, ...].astype(np.float32)

    # The real adapter exposes one source token for the selected aperture.
    # Keep field stars in the raster, but do not feed simulator-only catalog
    # fields (especially has_exoplanet) into a model trained on real tokens.
    source_xy = np.asarray(result["object_tokens"][0, :, 0, :2], dtype=np.float32)
    objects = np.zeros((1, source_xy.shape[0], 1, 12), dtype=np.float32)
    objects[0, :, 0, :2] = source_xy
    objects[0, :, 0, 2] = detector_width / canvas
    objects[0, :, 0, 3] = detector_height / canvas
    objects[0, :, 0, 10:] = 1.0
    result["object_tokens"] = objects
    result["object_mask"] = np.ones((1, source_xy.shape[0], 1), dtype=bool)
    result["exposure_duration"] = np.full_like(result["exposure_duration"], 1800.0)
    result["wavelength_tokens"][..., 0] = np.log10(650.0) / 4.0
    return result


def _compact_canvas_offset(
    sample_index: int,
    *,
    height: int,
    width: int,
    canvas: int = 32,
) -> tuple[int, int]:
    """Choose a deterministic non-centered compact-detector canvas offset."""

    available_y = canvas - height + 1
    available_x = canvas - width + 1
    rng = np.random.default_rng(97_531 + sample_index * 2_654_435_761)
    y0 = int(rng.integers(0, available_y))
    x0 = int(rng.integers(0, available_x))
    centered_y = (canvas - height) // 2
    centered_x = (canvas - width) // 2
    if available_y > 1 and y0 == centered_y:
        y0 = (y0 + 1) % available_y
    if available_x > 1 and x0 == centered_x:
        x0 = (x0 + 1) % available_x
    return y0, x0


def _embed_compact_view(
    view: dict[str, np.ndarray],
    *,
    sample_index: int,
    canvas: int = 32,
) -> dict[str, np.ndarray]:
    """Place a compact detector cutout on the model's fixed-size canvas."""

    raster = view["raster"]
    if raster.ndim != 6:
        raise ValueError(f"expected batched raster, got {raster.shape}")
    _, visits, steps, channels, height, width = raster.shape
    if height > canvas or width > canvas:
        raise ValueError(f"compact detector {height}x{width} does not fit {canvas}x{canvas}")
    if (height, width) == (canvas, canvas):
        return view
    y0, x0 = _compact_canvas_offset(
        sample_index,
        height=height,
        width=width,
        canvas=canvas,
    )
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
    *,
    transit_radius_ratio_min: float = 0.006,
    transit_radius_ratio_max: float = 0.04,
) -> tuple[list[dict[str, np.ndarray]], list[float], list[tuple[float, float]]]:
    """Generate one null/injected pair with an isolated deterministic seed."""

    source_x, source_y = _source_position(generator_config, sample_index)
    bundle_config = replace(
        _sample_target_event_config(
            generator_config,
            sample_index,
            transit_radius_ratio_min=transit_radius_ratio_min,
            transit_radius_ratio_max=transit_radius_ratio_max,
        ),
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
            bundle.as_model_numpy("null" if label == 0.0 else "injected"),
            sample_index=sample_index,
        )
        # The real Kepler adapter exposes channel 4 as a good-quality mask,
        # whereas the generator stores an interpolation mask. Convert the
        # latter to the former at this adapter boundary.
        raster = embedded["raster"]
        wavelength_tokens = embedded["wavelength_tokens"]
        raster[:, :, :, 4] = raster[:, :, :, 3] * (1.0 - raster[:, :, :, 4])
        embedded["raster"] = raster
        embedded = _match_real_adapter_context(
            embedded,
            timestamps_mid_bjd_tdb=bundle.timestamps_mid_bjd_tdb,
            detector_height=generator_config.raster_height,
            detector_width=generator_config.raster_width,
        )
        cadence_values = wavelength_tokens[..., 1]
        cadence_uncertainty = wavelength_tokens[..., 3]
        wavelength_tokens[..., 6] = _robust_temporal_score(
            cadence_values,
            cadence_uncertainty,
        )
        embedded["wavelength_tokens"] = wavelength_tokens
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
    *,
    transit_radius_ratio_min: float = 0.006,
    transit_radius_ratio_max: float = 0.04,
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
                lambda index: _generate_pair(
                    generator_config,
                    index,
                    transit_radius_ratio_min=transit_radius_ratio_min,
                    transit_radius_ratio_max=transit_radius_ratio_max,
                ),
                pair_indices,
            )
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            generated_pairs = executor.map(
                lambda index: _generate_pair(
                    generator_config,
                    index,
                    transit_radius_ratio_min=transit_radius_ratio_min,
                    transit_radius_ratio_max=transit_radius_ratio_max,
                ),
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


def _weighted_event_loss(
    prediction: dict[str, torch.Tensor],
    batch: AstroMambaHTrainingBatch,
    *,
    positive_weight: float,
    negative_weight: float,
) -> torch.Tensor:
    """Event BCE with explicit class weights for false-positive pressure."""

    target = batch.target.to(dtype=torch.float32)
    logits = prediction["global_event_logits"]
    weights = torch.where(
        target.reshape_as(logits) > 0.5,
        torch.full_like(logits, positive_weight),
        torch.full_like(logits, negative_weight),
    )
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target.reshape_as(logits),
        weight=weights,
    )


def _paired_ranking_loss(
    prediction: dict[str, torch.Tensor],
    batch: AstroMambaHTrainingBatch,
    *,
    margin: float,
) -> torch.Tensor:
    """Rank each injected target-transit view above its matched null view."""

    return _paired_ranking_loss_for_logits(
        prediction["global_event_logits"],
        batch.target,
        margin=margin,
    )


def _paired_ranking_loss_for_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Rank paired synthetic logits with shared null/target ordering."""

    logits = logits.reshape(-1)
    targets = targets.reshape(-1).to(dtype=torch.float32, device=logits.device)
    if logits.numel() < 2:
        return logits.new_zeros(())
    pair_count = logits.numel() // 2
    paired_logits = logits[: pair_count * 2].reshape(pair_count, 2)
    paired_targets = targets[: pair_count * 2].reshape(pair_count, 2)
    null_mask = paired_targets[:, 0] < 0.5
    positive_mask = paired_targets[:, 1] > 0.5
    valid_pairs = null_mask & positive_mask
    if not bool(valid_pairs.any()):
        return logits.new_zeros(())
    null_logits = paired_logits[valid_pairs, 0]
    positive_logits = paired_logits[valid_pairs, 1]
    return torch.nn.functional.softplus(margin - (positive_logits - null_logits)).mean()


def _auxiliary_event_head_loss(
    prediction: dict[str, torch.Tensor],
    batch: AstroMambaHTrainingBatch,
) -> torch.Tensor:
    """Directly supervise compact event evidence heads used by the global logit."""

    target = batch.target.to(dtype=torch.float32)
    terms: list[torch.Tensor] = []
    for name in (
        "source_photometry_event_logits",
        "source_dip_event_logits",
        "temporal_multiscale_event_logits",
        "temporal_feature_fusion_event_logits",
        "pooled_backbone_event_logits",
    ):
        logits = prediction.get(name)
        if isinstance(logits, torch.Tensor) and logits.shape == target.reshape_as(logits).shape:
            terms.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    target.reshape_as(logits),
                )
            )
    source_event_logits = prediction.get("source_event_logits")
    if isinstance(source_event_logits, torch.Tensor) and source_event_logits.ndim == 2:
        direct_source_logits = source_event_logits[:, :1]
        terms.append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                direct_source_logits,
                target.reshape_as(direct_source_logits),
            )
        )
    if not terms:
        return target.new_zeros(())
    return torch.stack(terms).mean()


def _auxiliary_paired_ranking_loss(
    prediction: dict[str, torch.Tensor],
    batch: AstroMambaHTrainingBatch,
    *,
    margin: float,
) -> torch.Tensor:
    """Pair-rank each compact event evidence head used by the global logit."""

    targets = batch.target
    terms: list[torch.Tensor] = []
    for name in (
        "source_photometry_event_logits",
        "source_dip_event_logits",
        "temporal_multiscale_event_logits",
        "temporal_feature_fusion_event_logits",
        "pooled_backbone_event_logits",
    ):
        logits = prediction.get(name)
        if isinstance(logits, torch.Tensor) and logits.reshape(-1).numel() == targets.reshape(-1).numel():
            terms.append(_paired_ranking_loss_for_logits(logits, targets, margin=margin))
    source_event_logits = prediction.get("source_event_logits")
    if isinstance(source_event_logits, torch.Tensor) and source_event_logits.ndim == 2:
        direct_source_logits = source_event_logits[:, :1]
        if direct_source_logits.reshape(-1).numel() == targets.reshape(-1).numel():
            terms.append(_paired_ranking_loss_for_logits(direct_source_logits, targets, margin=margin))
    if not terms:
        return targets.new_zeros(())
    return torch.stack(terms).mean()


def _new_model(device: torch.device) -> AstroMambaHTrainingAdapter:
    """Create FP32 master weights for a clean post-contract curriculum."""

    model = AstroMambaHTrainingAdapter(_config()).to(device, dtype=torch.float32)
    model._checkpoint_transfer = {"mode": "from_scratch", "compatible_tensors": 0}
    return model


def _freeze_except_event_heads(model: AstroMambaHTrainingAdapter) -> None:
    """Keep the learned representation fixed during synthetic recalibration."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (
        model.core.temporal_summary_event,
        model.core.temporal_shape_event,
        model.core.temporal_robust_event,
        model.core.temporal_matched_event,
        model.core.temporal_sequence_event,
        model.core.temporal_sequence_projection,
        model.core.temporal_multiscale_event,
        model.core.temporal_multiscale_projection,
        model.core.temporal_feature_fusion_event,
        model.core.source_dip_event,
        model.core.source_photometry_projection,
        model.core.source_photometry_event,
        model.core.event_evidence_calibration,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    for parameter in (
        model.core.event_logit_scale,
        model.core.event_source_weight,
        model.core.event_backbone_weight,
        model.core.event_photometry_weight,
    ):
        parameter.requires_grad = True


def _freeze_except_event_calibration(model: AstroMambaHTrainingAdapter) -> None:
    """Train only the bounded global event calibration parameters and head."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in (
        model.core.event_logit_scale,
        model.core.event_source_weight,
        model.core.event_backbone_weight,
        model.core.event_photometry_weight,
    ):
        parameter.requires_grad = True
    for parameter in model.core.event_evidence_calibration.parameters():
        parameter.requires_grad = True


def _reset_source_photometry_branch(model: AstroMambaHTrainingAdapter) -> None:
    """Reinitialize the compact source-photometry branch after feature shifts."""

    for module in (model.core.source_photometry_projection, model.core.source_photometry_event):
        module.apply(lambda child: child.reset_parameters() if hasattr(child, "reset_parameters") else None)


def _zero_event_photometry_weight(model: AstroMambaHTrainingAdapter) -> None:
    """Disable the direct photometry shortcut while preserving calibrator evidence."""

    with torch.no_grad():
        model.core.event_photometry_weight.zero_()


def _reset_temporal_event_heads(model: AstroMambaHTrainingAdapter) -> None:
    """Reinitialize compact temporal event heads while preserving the backbone."""

    modules = (
        model.core.temporal_summary_event,
        model.core.temporal_shape_event,
        model.core.temporal_robust_event,
        model.core.temporal_matched_event,
        model.core.temporal_sequence_event,
        model.core.temporal_sequence_projection,
        model.core.temporal_multiscale_event,
        model.core.temporal_multiscale_projection,
        model.core.temporal_feature_fusion_event,
        model.core.source_dip_event,
        model.core.event_evidence_calibration,
    )
    for module in modules:
        module.apply(lambda child: child.reset_parameters() if hasattr(child, "reset_parameters") else None)
    for module in (
        model.core.temporal_summary_event,
        model.core.temporal_shape_event,
        model.core.temporal_robust_event,
        model.core.temporal_matched_event,
        model.core.temporal_sequence_projection,
        model.core.temporal_multiscale_projection,
        model.core.temporal_feature_fusion_event,
        model.core.source_dip_event,
        model.core.event_evidence_calibration,
    ):
        final = module[-1]
        final.weight.data.zero_()
        final.bias.data.zero_()
    with torch.no_grad():
        model.core.event_logit_scale.fill_(1.0)
        model.core.event_source_weight.fill_(1.0)
        model.core.event_backbone_weight.fill_(0.5)
        model.core.event_photometry_weight.fill_(0.5)


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
    *,
    transit_radius_ratio_min: float = 0.006,
    transit_radius_ratio_max: float = 0.04,
    progress_label: str | None = None,
    progress_epoch: int | None = None,
    progress_log_frequency: int = 0,
) -> dict[str, object]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    batches_total = math.ceil(pair_count / batch_pairs)
    with torch.inference_mode():
        for offset in range(0, pair_count, batch_pairs):
            count = min(batch_pairs, pair_count - offset)
            batch = _make_batch(
                generator_config,
                start_index + offset,
                count,
                device,
                _batch_cache_path(cache_dir, start_index + offset, count),
                transit_radius_ratio_min=transit_radius_ratio_min,
                transit_radius_ratio_max=transit_radius_ratio_max,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
            probabilities.extend(float(value) for value in output["global_event_logits"].float().sigmoid().cpu())
            labels.extend([int(value) for value in batch.target.reshape(-1).cpu()])
            del batch, output
            batches_done = offset // batch_pairs + 1
            if progress_label and progress_log_frequency and batches_done % progress_log_frequency == 0:
                print(
                    json.dumps(
                        {
                            "event": "eval_batch",
                            "label": progress_label,
                            "epoch": progress_epoch,
                            "batches_done": batches_done,
                            "batches_total": batches_total,
                            "samples_done": len(labels),
                            "samples_total": pair_count * 2,
                            "rss_bytes": _rss(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    predicted = [int(value >= 0.5) for value in probabilities]
    correct = sum(prediction == label for prediction, label in zip(predicted, labels))
    best_threshold = 0.5
    best_threshold_correct = correct
    for threshold in sorted(set(probabilities)):
        threshold_predictions = [int(value >= threshold) for value in probabilities]
        threshold_correct = sum(
            prediction == label
            for prediction, label in zip(threshold_predictions, labels)
        )
        if threshold_correct > best_threshold_correct:
            best_threshold = float(threshold)
            best_threshold_correct = threshold_correct
    positives = [probability for probability, label in zip(probabilities, labels) if label == 1]
    negatives = [probability for probability, label in zip(probabilities, labels) if label == 0]
    pairwise_total = len(positives) * len(negatives)
    pairwise_wins = 0.0
    if pairwise_total:
        for positive in positives:
            for negative in negatives:
                if positive > negative:
                    pairwise_wins += 1.0
                elif positive == negative:
                    pairwise_wins += 0.5
    return {
        "pairs": pair_count,
        "samples": len(labels),
        "correct": correct,
        "errors": len(labels) - correct,
        "accuracy": correct / len(labels) if labels else None,
        "best_threshold": best_threshold,
        "best_threshold_correct": best_threshold_correct,
        "best_threshold_errors": len(labels) - best_threshold_correct,
        "best_threshold_accuracy": best_threshold_correct / len(labels) if labels else None,
        "probability_auc": pairwise_wins / pairwise_total if pairwise_total else None,
        "mean_positive_probability": sum(positives) / len(positives) if positives else None,
        "mean_negative_probability": sum(negatives) / len(negatives) if negatives else None,
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
        "--positive-loss-weight",
        type=float,
        default=1.0,
        help="event-loss weight for injected target-transit examples",
    )
    parser.add_argument(
        "--negative-loss-weight",
        type=float,
        default=1.0,
        help="event-loss weight for null and off-target-only examples",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("event", "source"),
        default="event",
        help="event-only detector curriculum, or proposal-aware source auxiliary loss",
    )
    parser.add_argument(
        "--train-only-event-heads",
        action="store_true",
        help="freeze the backbone and adapt only compact event decision heads",
    )
    parser.add_argument(
        "--train-only-event-calibration",
        action="store_true",
        help="freeze feature heads and train only bounded global event calibration",
    )
    parser.add_argument(
        "--reset-source-photometry-branch",
        action="store_true",
        help="reinitialize the compact source-photometry branch after loading a checkpoint",
    )
    parser.add_argument(
        "--zero-event-photometry-weight",
        action="store_true",
        help="zero the direct photometry contribution after loading a checkpoint",
    )
    parser.add_argument(
        "--reset-temporal-event-heads",
        action="store_true",
        help="reinitialize compact temporal event heads after loading a checkpoint",
    )
    parser.add_argument(
        "--paired-ranking-weight",
        type=float,
        default=0.0,
        help="optional weight for ranking each injected target-transit view above its matched null view",
    )
    parser.add_argument(
        "--paired-ranking-margin",
        type=float,
        default=0.5,
        help="minimum desired injected-minus-null logit margin for paired synthetic ranking",
    )
    parser.add_argument(
        "--auxiliary-event-head-weight",
        type=float,
        default=0.0,
        help="optional BCE weight for directly supervising compact event evidence heads",
    )
    parser.add_argument(
        "--auxiliary-paired-ranking-weight",
        type=float,
        default=0.0,
        help="optional pair-ranking weight for compact event evidence heads",
    )
    parser.add_argument(
        "--training-eval-frequency",
        type=int,
        default=1,
        help=(
            "score the full synthetic training set every N epochs; new-best, final, "
            "and early-stop epochs are still scored"
        ),
    )
    parser.add_argument(
        "--defer-training-eval-until-final",
        action="store_true",
        help=(
            "skip full synthetic training-set evaluation during improving epochs; "
            "the selected best checkpoint is still scored once before writing the final report"
        ),
    )
    parser.add_argument(
        "--progress-log-frequency",
        type=int,
        default=0,
        help="emit lightweight JSON progress every N training batches; 0 disables batch progress logs",
    )
    parser.add_argument(
        "--field-star-count",
        type=int,
        help="override the number of background/neighbor stars in each synthetic cutout",
    )
    parser.add_argument(
        "--field-planet-probability",
        type=float,
        help="override per-neighbor probability of an off-target transit-bearing planet",
    )
    parser.add_argument(
        "--stellar-brightness-noise-sigma",
        type=float,
        help="override target-star brightness fluctuation sigma for staged synthetic curricula",
    )
    parser.add_argument(
        "--transit-radius-ratio-min",
        type=float,
        default=0.006,
        help="minimum sampled target transit radius ratio",
    )
    parser.add_argument(
        "--transit-radius-ratio-max",
        type=float,
        default=0.04,
        help="maximum sampled target transit radius ratio",
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
    if args.positive_loss_weight <= 0.0 or args.negative_loss_weight <= 0.0:
        raise ValueError("loss weights must be positive")
    if args.paired_ranking_weight < 0.0:
        raise ValueError("paired-ranking-weight must be non-negative")
    if args.paired_ranking_margin < 0.0:
        raise ValueError("paired-ranking-margin must be non-negative")
    if args.auxiliary_event_head_weight < 0.0:
        raise ValueError("auxiliary-event-head-weight must be non-negative")
    if args.auxiliary_paired_ranking_weight < 0.0:
        raise ValueError("auxiliary-paired-ranking-weight must be non-negative")
    if args.training_eval_frequency < 1:
        raise ValueError("training-eval-frequency must be positive")
    if args.progress_log_frequency < 0:
        raise ValueError("progress-log-frequency must be non-negative")
    if args.field_star_count is not None and args.field_star_count < 0:
        raise ValueError("field-star-count must be non-negative")
    if args.field_planet_probability is not None and not 0.0 <= args.field_planet_probability <= 1.0:
        raise ValueError("field-planet-probability must be in [0, 1]")
    if args.stellar_brightness_noise_sigma is not None and args.stellar_brightness_noise_sigma < 0.0:
        raise ValueError("stellar-brightness-noise-sigma must be non-negative")
    if args.transit_radius_ratio_min < 0.0 or args.transit_radius_ratio_max < 0.0:
        raise ValueError("transit radius ratio bounds must be non-negative")
    if args.transit_radius_ratio_min > args.transit_radius_ratio_max:
        raise ValueError("transit-radius-ratio-min cannot exceed transit-radius-ratio-max")
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
    if args.reset_source_photometry_branch:
        _reset_source_photometry_branch(model)
    if args.zero_event_photometry_weight:
        _zero_event_photometry_weight(model)
    if args.reset_temporal_event_heads:
        _reset_temporal_event_heads(model)
    if args.train_only_event_heads:
        _freeze_except_event_heads(model)
    if args.train_only_event_calibration:
        _freeze_except_event_calibration(model)
    if args.loss_mode == "event" and (
        args.positive_loss_weight != 1.0 or args.negative_loss_weight != 1.0
    ):
        loss_fn = lambda output, batch: _weighted_event_loss(
            output,
            batch,
            positive_weight=args.positive_loss_weight,
            negative_weight=args.negative_loss_weight,
        )
    else:
        loss_fn = event_only_loss_fn if args.loss_mode == "event" else source_event_loss_fn
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("event-head-only mode produced no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.learning_rate, weight_decay=1e-4)
    generator_config = _apply_curriculum_overrides(
        _synthetic_config(seed=23),
        field_star_count=args.field_star_count,
        field_planet_probability=args.field_planet_probability,
        stellar_brightness_noise_sigma=args.stellar_brightness_noise_sigma,
    )
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
    best_epoch = None
    best_epoch_checkpoint = None
    best_epoch_mean_loss = None
    best_training = None
    best_holdout = None
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
                transit_radius_ratio_min=args.transit_radius_ratio_min,
                transit_radius_ratio_max=args.transit_radius_ratio_max,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
                loss = loss_fn(output, batch)
                if args.loss_mode == "source" and (
                    args.positive_loss_weight != 1.0 or args.negative_loss_weight != 1.0
                ):
                    loss = loss + _weighted_event_loss(
                        output,
                        batch,
                        positive_weight=args.positive_loss_weight,
                        negative_weight=args.negative_loss_weight,
                    ) - event_only_loss_fn(output, batch)
                if args.paired_ranking_weight > 0.0:
                    loss = loss + args.paired_ranking_weight * _paired_ranking_loss(
                        output,
                        batch,
                        margin=args.paired_ranking_margin,
                    )
                if args.auxiliary_event_head_weight > 0.0:
                    loss = loss + args.auxiliary_event_head_weight * _auxiliary_event_head_loss(
                        output,
                        batch,
                    )
                if args.auxiliary_paired_ranking_weight > 0.0:
                    loss = loss + args.auxiliary_paired_ranking_weight * _auxiliary_paired_ranking_loss(
                        output,
                        batch,
                        margin=args.paired_ranking_margin,
                    )
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
            if args.progress_log_frequency and global_step % args.progress_log_frequency == 0:
                print(
                    json.dumps(
                        {
                            "event": "train_batch",
                            "epoch": epoch,
                            "global_step": global_step,
                            "epoch_batches_done": len(losses),
                            "epoch_batches_total": math.ceil(args.pair_count / args.batch_pairs),
                            "mean_loss_so_far": sum(losses) / len(losses),
                            "rss_bytes": peak_rss,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        final_holdout = _evaluate(
            model,
            generator_config,
            args.holdout_start_index,
            args.holdout_pairs,
            args.batch_pairs,
            device,
            cache_dir,
            transit_radius_ratio_min=args.transit_radius_ratio_min,
            transit_radius_ratio_max=args.transit_radius_ratio_max,
            progress_label="holdout",
            progress_epoch=epoch,
            progress_log_frequency=args.progress_log_frequency,
        )
        synthetic_improved = best_holdout is None or final_holdout["errors"] < best_holdout["errors"]
        should_stop = (
            args.stop_holdout_errors is not None
            and final_holdout["errors"] <= args.stop_holdout_errors
        )
        should_score_training = (
            (synthetic_improved and not args.defer_training_eval_until_final)
            or should_stop
            or epoch + 1 == args.max_epochs
            or (epoch + 1) % args.training_eval_frequency == 0
        )
        final_training = (
            _evaluate(
                model,
                generator_config,
                args.start_index,
                args.pair_count,
                args.batch_pairs,
                device,
                cache_dir,
                transit_radius_ratio_min=args.transit_radius_ratio_min,
                transit_radius_ratio_max=args.transit_radius_ratio_max,
                progress_label="training",
                progress_epoch=epoch,
                progress_log_frequency=args.progress_log_frequency,
            )
            if should_score_training
            else None
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
                "positive_loss_weight": args.positive_loss_weight,
                "negative_loss_weight": args.negative_loss_weight,
                "paired_ranking_weight": args.paired_ranking_weight,
                "paired_ranking_margin": args.paired_ranking_margin,
                "auxiliary_event_head_weight": args.auxiliary_event_head_weight,
                "auxiliary_paired_ranking_weight": args.auxiliary_paired_ranking_weight,
                "training_eval_frequency": args.training_eval_frequency,
                "defer_training_eval_until_final": args.defer_training_eval_until_final,
                "progress_log_frequency": args.progress_log_frequency,
                "reset_source_photometry_branch": args.reset_source_photometry_branch,
                "zero_event_photometry_weight": args.zero_event_photometry_weight,
                "reset_temporal_event_heads": args.reset_temporal_event_heads,
                "train_only_event_heads": args.train_only_event_heads,
                "train_only_event_calibration": args.train_only_event_calibration,
                "scene_model": "multi_star_field_target_counterfactual",
                "field_star_count": generator_config.field_star_count,
                "field_planet_probability": generator_config.field_planet_probability,
                "stellar_brightness_noise_sigma": generator_config.stellar_brightness_noise_sigma,
                "transit_radius_ratio_min": args.transit_radius_ratio_min,
                "transit_radius_ratio_max": args.transit_radius_ratio_max,
                "source_position_policy": "deterministic_noncentral_source_inside_compact_detector_plus_noncentered_canvas_offset",
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
        if synthetic_improved:
            if final_training is None and not args.defer_training_eval_until_final:
                final_training = _evaluate(
                    model,
                    generator_config,
                    args.start_index,
                    args.pair_count,
                    args.batch_pairs,
                    device,
                    cache_dir,
                    transit_radius_ratio_min=args.transit_radius_ratio_min,
                    transit_radius_ratio_max=args.transit_radius_ratio_max,
                    progress_label="training",
                    progress_epoch=epoch,
                    progress_log_frequency=args.progress_log_frequency,
                )
                record["training"] = final_training
            best_epoch = epoch
            best_epoch_checkpoint = epoch_checkpoint
            best_epoch_mean_loss = record["mean_loss"]
            best_training = final_training
            best_holdout = final_holdout
        if (final_training is not None and final_training["errors"] == 0) or should_stop:
            break
    if (
        best_holdout is None
        or best_epoch_checkpoint is None
        or best_epoch is None
    ):
        raise RuntimeError("no synthetic epoch completed")
    if best_training is None:
        best_state = torch.load(best_epoch_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model"], strict=False)
        best_training = _evaluate(
            model,
            generator_config,
            args.start_index,
            args.pair_count,
            args.batch_pairs,
            device,
            cache_dir,
            transit_radius_ratio_min=args.transit_radius_ratio_min,
            transit_radius_ratio_max=args.transit_radius_ratio_max,
            progress_label="training",
            progress_epoch=best_epoch,
            progress_log_frequency=args.progress_log_frequency,
        )
    final_holdout = best_holdout
    final_training = best_training
    temporary = args.output_checkpoint.with_suffix(args.output_checkpoint.suffix + ".tmp")
    shutil.copy2(best_epoch_checkpoint, temporary)
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
        "positive_loss_weight": args.positive_loss_weight,
        "negative_loss_weight": args.negative_loss_weight,
        "paired_ranking_weight": args.paired_ranking_weight,
        "paired_ranking_margin": args.paired_ranking_margin,
        "auxiliary_event_head_weight": args.auxiliary_event_head_weight,
        "auxiliary_paired_ranking_weight": args.auxiliary_paired_ranking_weight,
        "training_eval_frequency": args.training_eval_frequency,
        "defer_training_eval_until_final": args.defer_training_eval_until_final,
        "progress_log_frequency": args.progress_log_frequency,
        "reset_source_photometry_branch": args.reset_source_photometry_branch,
        "zero_event_photometry_weight": args.zero_event_photometry_weight,
        "reset_temporal_event_heads": args.reset_temporal_event_heads,
        "train_only_event_heads": args.train_only_event_heads,
        "train_only_event_calibration": args.train_only_event_calibration,
        "field_star_count": generator_config.field_star_count,
        "field_planet_probability": generator_config.field_planet_probability,
        "stellar_brightness_noise_sigma": generator_config.stellar_brightness_noise_sigma,
        "transit_radius_ratio_min": args.transit_radius_ratio_min,
        "transit_radius_ratio_max": args.transit_radius_ratio_max,
        "source_position_policy": "deterministic_noncentral_source_inside_compact_detector_plus_noncentered_canvas_offset",
        "cache_dir": str(cache_dir),
        "cache_enabled": True,
        "sample_index_first": args.start_index,
        "sample_index_last": args.start_index + args.pair_count - 1,
        "holdout_start_index": args.holdout_start_index,
        "holdout": final_holdout,
        "training": final_training,
        "selected_epoch": best_epoch,
        "selected_epoch_checkpoint": str(best_epoch_checkpoint),
        "selected_epoch_mean_loss": best_epoch_mean_loss,
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
