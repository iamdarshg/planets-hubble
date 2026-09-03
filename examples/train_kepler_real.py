"""Train AstroMamba-H on observed Kepler target-pixel-file windows.

This is a real-data adapter, not the synthetic injection path.  Each record is
an observed MAST TPF window labelled from the Kepler DR25 confirmed-planet
ephemeris.  The compact TPF pixels are placed at a deterministic, host-keyed
non-centered offset in a 32x32 canvas by padding only; no spatial resampling or
synthetic transit generation is performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import AstroMambaHConfig  # noqa: E402
from training.adapters import (  # noqa: E402
    AstroMambaHInputs,
    AstroMambaHTrainingAdapter,
    AstroMambaHTrainingBatch,
)
from training.harness import event_only_loss_fn  # noqa: E402


RSS_CAP_BYTES = 840_000_000


def _records(manifest_path: Path) -> list[dict[str, object]]:
    root = manifest_path.parent
    records: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            path = root / Path(str(record["path"]))
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(record)
    if not records:
        raise ValueError("real corpus manifest is empty")
    return records


def _batch_records(records: list[dict[str, object]], batch_size: int, *, seed: int, shuffle: bool) -> Iterable[list[dict[str, object]]]:
    indices = np.arange(len(records))
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [records[int(index)] for index in indices[start : start + batch_size]]


def _pair_key(record: dict[str, object]) -> str:
    stem = Path(str(record["path"])).stem
    return stem.rsplit("-", 1)[0]


def _canvas_offset(record: dict[str, object], height: int, width: int, canvas: int) -> tuple[int, int]:
    """Choose a stable non-centered offset shared by a host's pair.

    The offset is derived only from the TPF identity, not the label or the
    event/control center.  This preserves the counterfactual pair contract
    while exposing the spatial encoder to the same translation variation as
    the synthetic observations.
    """

    if height > canvas or width > canvas:
        raise ValueError(f"TPF cutout {height}x{width} does not fit compact {canvas}x{canvas} canvas")
    provenance = record.get("provenance")
    tpf_url = provenance.get("tpf_url") if isinstance(provenance, dict) else None
    identity = tpf_url if isinstance(tpf_url, str) and tpf_url else _pair_key(record)
    digest = hashlib.blake2b(identity.encode("utf-8"), digest_size=8).digest()
    available_y = canvas - height + 1
    available_x = canvas - width + 1
    y0 = int.from_bytes(digest[:4], "little") % available_y
    x0 = int.from_bytes(digest[4:], "little") % available_x
    centered_y = (canvas - height) // 2
    centered_x = (canvas - width) // 2
    if available_y > 1 and y0 == centered_y:
        y0 = (y0 + 1) % available_y
    if available_x > 1 and x0 == centered_x:
        x0 = (x0 + 1) % available_x
    return y0, x0


def _paired_batch_records(
    records: list[dict[str, object]], batch_size: int, *, seed: int
) -> Iterable[list[dict[str, object]]]:
    """Batch positive/control counterfactual windows from the same target."""

    if batch_size < 2:
        raise ValueError("paired ranking requires batch-size >= 2")
    grouped: dict[str, dict[int, dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(_pair_key(record), {})[int(record["label"])] = record
    pairs = [
        (items[1], items[0])
        for items in grouped.values()
        if 0 in items and 1 in items
    ]
    if not pairs:
        raise ValueError("paired ranking found no complete positive/control pairs")
    rng = np.random.default_rng(seed)
    order = np.arange(len(pairs))
    rng.shuffle(order)
    pairs_per_batch = max(1, batch_size // 2)
    for start in range(0, len(order), pairs_per_batch):
        batch: list[dict[str, object]] = []
        for index in order[start : start + pairs_per_batch]:
            positive, control = pairs[int(index)]
            batch.extend((positive, control))
        yield batch


def _normalize_frame_arrays(
    science: np.ndarray,
    uncertainty: np.ndarray,
    finite: np.ndarray,
    quality: np.ndarray,
    *,
    canvas: int = 32,
    aperture_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    if science.ndim != 3 or science.shape != uncertainty.shape or science.shape != finite.shape:
        raise ValueError(f"invalid TPF arrays: science={science.shape} uncertainty={uncertainty.shape} finite={finite.shape}")
    frames, height, width = science.shape
    if height > canvas or width > canvas:
        raise ValueError(f"TPF cutout {height}x{width} does not fit compact {canvas}x{canvas} canvas")
    if not 0.0 < aperture_fraction <= 1.0:
        raise ValueError("aperture_fraction must be in (0, 1]")
    valid = finite.astype(bool) & np.isfinite(science)
    positive_values = science[valid & (science > 0.0)]
    if positive_values.size == 0:
        raise ValueError("TPF has no positive finite detector values")
    baseline = float(np.median(positive_values))
    residual_values = science[valid] - baseline
    robust_scale = float(np.median(np.abs(residual_values))) * 1.4826
    uncertainty_values = np.abs(uncertainty[np.isfinite(uncertainty)])
    uncertainty_scale = float(np.percentile(uncertainty_values, 75)) if uncertainty_values.size else 0.0
    scale = max(robust_scale, uncertainty_scale, 1.0)

    normalized = np.zeros_like(science, dtype=np.float32)
    residual = np.zeros_like(science, dtype=np.float32)
    normalized[valid] = science[valid] / max(abs(baseline), 1.0) - 1.0
    residual[valid] = (science[valid] - baseline) / scale
    normalized = np.clip(normalized, -20.0, 20.0)
    residual = np.clip(residual, -20.0, 20.0)
    err = np.nan_to_num(np.abs(uncertainty) / scale, nan=0.0, posinf=20.0, neginf=0.0).astype(np.float32)
    err = np.clip(err, 0.0, 20.0)
    quality_ok = valid & (quality[:, None, None] == 0)

    # Use a static source-weighted aperture for the direct photometry path.
    # Kepler TPF cutouts include background pixels; summing every pixel lowers
    # transit SNR and makes the learned event branch depend on cutout size.
    # The aperture is selected from the time-median detector image, never from
    # labels or a transit-centered statistic.
    median_image = np.nanmedian(np.where(valid, science, np.nan), axis=0)
    finite_median = np.isfinite(median_image)
    if not finite_median.any():
        aperture_mask = np.ones((height, width), dtype=bool)
    else:
        count = max(1, int(np.ceil(float(finite_median.sum()) * aperture_fraction)))
        ranked = np.where(finite_median, median_image, -np.inf).reshape(-1)
        selected = np.argpartition(ranked, -count)[-count:]
        aperture_mask = np.zeros(height * width, dtype=bool)
        aperture_mask[selected] = True
        aperture_mask = aperture_mask.reshape(height, width)
    aperture_valid = valid & aperture_mask[None, :, :]
    aperture = np.sum(np.where(aperture_valid, science, 0.0), axis=(1, 2)).astype(np.float64)
    aperture_baseline = float(np.median(aperture[aperture > 0.0])) if np.any(aperture > 0.0) else 1.0
    photometry = np.clip(aperture / max(abs(aperture_baseline), 1.0) - 1.0, -20.0, 20.0).astype(np.float32)
    aperture_error = np.sqrt(
        np.sum(
            np.where(aperture_valid, np.square(np.nan_to_num(uncertainty, nan=0.0)), 0.0),
            axis=(1, 2),
        )
    )
    photometric_uncertainty = np.clip(aperture_error / max(abs(aperture_baseline), 1.0), 0.0, 20.0).astype(np.float32)
    return normalized, residual, err, valid.astype(np.float32), quality_ok.astype(np.float32), photometry, photometric_uncertainty


def _robust_temporal_score(photometry: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """Expose a scale-invariant cadence-level dip feature.

    The score is computed from the observed aperture sequence only. Using a
    robust scale prevents a single cosmic-ray/outlier cadence from erasing a
    shallow transit signature while preserving the sign of a flux dip.
    """

    center = float(np.median(photometry))
    mad_scale = float(np.median(np.abs(photometry - center))) * 1.4826
    scale = max(mad_scale, float(np.median(np.abs(uncertainty))), 1.0e-5)
    return np.clip((photometry - center) / scale, -20.0, 20.0).astype(np.float32)


@lru_cache(maxsize=16)
def _load_full_tpf_lightcurve(raw_tpf_dir: str, filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load one bounded full-quarter aperture light curve for local detrending."""

    path = Path(raw_tpf_dir) / filename
    if not path.is_file():
        return None
    try:
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdul:
            table = hdul[1].data
            times = np.asarray(table["TIME"], dtype=np.float64)
            flux = np.asarray(table["FLUX"], dtype=np.float32)
            if "FLUX_BKG" in table.names:
                background = np.asarray(table["FLUX_BKG"], dtype=np.float32)
                flux = flux - np.where(np.isfinite(background), background, 0.0)
            if not np.isfinite(flux).any():
                flux = np.asarray(table["RAW_CNTS"], dtype=np.float32)
        valid = np.isfinite(times) & np.isfinite(flux).any(axis=(1, 2))
        aperture = np.sum(np.where(np.isfinite(flux), flux, 0.0), axis=(1, 2)).astype(np.float64)
        valid &= aperture > 0.0
        if int(valid.sum()) < 8:
            return None
        finite_times = times[valid]
        finite_aperture = aperture[valid]
        center = float(np.median(finite_times))
        scaled_time = (finite_times - center) / max(float(np.ptp(finite_times)), 1.0)
        # A quarter contains enough valid cadences to support a modestly
        # higher-order baseline. Keep the order bounded so the fit follows
        # broad instrumental/stellar curvature without becoming a local
        # transit interpolator.
        degree = min(6, int(valid.sum()) - 1)
        coefficients = np.polyfit(scaled_time, finite_aperture, degree)
        baseline = np.full(times.shape, np.nan, dtype=np.float64)
        baseline[valid] = np.polyval(coefficients, scaled_time)
        baseline[valid] = np.maximum(baseline[valid], 1.0)
        return times, aperture, baseline, valid
    except (OSError, ValueError, KeyError, IndexError):
        return None


def _full_tpf_detrended_photometry(
    root: Path,
    record: dict[str, object],
    local_times: np.ndarray,
    *,
    raw_tpf_dir: Path,
) -> np.ndarray | None:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        return None
    tpf_url = provenance.get("tpf_url")
    if not isinstance(tpf_url, str) or not tpf_url:
        return None
    filename = tpf_url.rsplit("/", 1)[-1]
    loaded = _load_full_tpf_lightcurve(str(raw_tpf_dir.resolve()), filename)
    if loaded is None:
        return None
    full_times, aperture, baseline, valid = loaded
    indices = np.asarray(
        [int(np.nanargmin(np.abs(full_times - value))) for value in local_times],
        dtype=np.int64,
    )
    local_valid = valid[indices] & np.isfinite(baseline[indices])
    detrended = np.zeros(local_times.shape, dtype=np.float32)
    detrended[local_valid] = (
        aperture[indices[local_valid]] / baseline[indices[local_valid]] - 1.0
    ).astype(np.float32)
    return np.clip(detrended, -20.0, 20.0)


def _load_example(
    root: Path,
    record: dict[str, object],
    *,
    canvas: int = 32,
    aperture_fraction: float = 1.0,
    raw_tpf_dir: Path | None = None,
    full_tpf_detrend: bool = False,
) -> dict[str, np.ndarray | float | int | str]:
    with np.load(root / Path(str(record["path"])), allow_pickle=False) as arrays:
        science = np.asarray(arrays["science"], dtype=np.float32)
        uncertainty = np.asarray(arrays["uncertainty"], dtype=np.float32)
        finite = np.asarray(arrays["finite"], dtype=np.uint8)
        quality = np.asarray(arrays["quality"], dtype=np.int32)
        times = np.asarray(arrays["time"], dtype=np.float64)
    normalized, residual, err, valid, quality_ok, photometry, photometric_uncertainty = _normalize_frame_arrays(
        science,
        uncertainty,
        finite,
        quality,
        canvas=canvas,
        aperture_fraction=aperture_fraction,
    )
    temporal_score_photometry = photometry
    if full_tpf_detrend and raw_tpf_dir is not None:
        detrended = _full_tpf_detrended_photometry(
            root,
            record,
            times,
            raw_tpf_dir=raw_tpf_dir,
        )
        if detrended is not None:
            temporal_score_photometry = detrended
    temporal_score = _robust_temporal_score(
        temporal_score_photometry,
        photometric_uncertainty,
    )
    frames, height, width = normalized.shape
    raster = np.zeros((frames, 6, canvas, canvas), dtype=np.float32)
    y0, x0 = _canvas_offset(record, height, width, canvas)
    raster[:, 0, y0 : y0 + height, x0 : x0 + width] = normalized
    raster[:, 1, y0 : y0 + height, x0 : x0 + width] = residual
    raster[:, 2, y0 : y0 + height, x0 : x0 + width] = err
    raster[:, 3, y0 : y0 + height, x0 : x0 + width] = valid
    raster[:, 4, y0 : y0 + height, x0 : x0 + width] = quality_ok
    raster[:, 5, y0 : y0 + height, x0 : x0 + width] = valid

    finite_times = np.isfinite(times)
    if finite_times.sum() < 1:
        raise ValueError("TPF window has no finite times")
    center = float(np.median(times[finite_times]))
    cadence_days = float(np.median(np.diff(times[finite_times]))) if finite_times.sum() > 1 else 0.0204
    cadence_seconds = cadence_days * 86400.0
    valid_fraction = valid.mean(axis=(1, 2)).astype(np.float32)
    quality_fraction = quality_ok.mean(axis=(1, 2)).astype(np.float32)
    local_time = np.stack(
        (
            times - center,
            times - center,
            times - center,
            np.full(frames, cadence_days, dtype=np.float64),
            np.ones(frames, dtype=np.float64),
        ),
        axis=-1,
    ).astype(np.float32)
    wavelength_tokens = np.stack(
        (
            np.full(frames, np.log10(650.0) / 4.0, dtype=np.float32),
            # SyntheticGenerator and the real adapter share the same first
            # three photometry features: ratio-like flux, bounded residual,
            # and normalized uncertainty. The previous real path supplied a
            # zero-centered ratio followed by uncertainty, which made the
            # checkpoint see a different modality ordering at fine-tuning.
            1.0 + photometry,
            np.clip(
                photometry / np.maximum(photometric_uncertainty, 1.0e-3),
                -20.0,
                20.0,
            ),
            photometric_uncertainty,
            # Keep feature 4 as the generator's normalized bandwidth slot.
            # The source-conditioned event branch consumes feature 6 for the
            # robust cadence-level dip score in both synthetic and real data.
            np.full(frames, 80.0 / 650.0, dtype=np.float32),
            np.full(frames, np.log10(max(cadence_seconds, 1.0)) / 5.0, dtype=np.float32),
            temporal_score,
            # Match the synthetic validity-mask semantics. Quality flags are
            # retained as a separate coverage feature; they do not erase an
            # otherwise finite detector measurement from the photometry path.
            valid_fraction,
        ),
        axis=-1,
    )[:, None, :]
    coverage = np.stack(
        (
            np.ones(frames, dtype=np.float32),
            valid_fraction,
            quality_fraction,
            np.full(frames, height / canvas, dtype=np.float32),
            np.full(frames, width / canvas, dtype=np.float32),
            np.ones(frames, dtype=np.float32),
        ),
        axis=-1,
    )
    long_time = np.asarray(
        [float(times.min() - center), float(times.max() - center), float(frames), float(times.max() - times.min()), 1.0],
        dtype=np.float32,
    )
    object_tokens = np.asarray(
        [[(x0 + width / 2.0) / canvas, (y0 + height / 2.0) / canvas, width / canvas, height / canvas, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]],
        dtype=np.float32,
    )
    return {
        "raster": raster,
        "wavelength_tokens": wavelength_tokens,
        "geometry": np.zeros((frames, 10), dtype=np.float32),
        "coverage": coverage,
        "local_time": local_time,
        "long_time": long_time,
        "object_tokens": object_tokens,
        "times": times,
        "label": int(record["label"]),
        "target_name": str(record["target_name"]),
    }


def _make_batch(
    root: Path,
    records: list[dict[str, object]],
    device: torch.device,
    *,
    aperture_fraction: float = 1.0,
    raw_tpf_dir: Path | None = None,
    full_tpf_detrend: bool = False,
) -> AstroMambaHTrainingBatch:
    examples = [
        _load_example(
            root,
            record,
            aperture_fraction=aperture_fraction,
            raw_tpf_dir=raw_tpf_dir,
            full_tpf_detrend=full_tpf_detrend,
        )
        for record in records
    ]
    batch_size = len(examples)
    frames = max(int(example["raster"].shape[0]) for example in examples)  # type: ignore[union-attr]
    raster = np.zeros((batch_size, 1, frames, 6, 32, 32), dtype=np.float32)
    wavelength = np.zeros((batch_size, 1, frames, 1, 8), dtype=np.float32)
    geometry = np.zeros((batch_size, 1, frames, 10), dtype=np.float32)
    coverage = np.zeros((batch_size, 1, frames, 6), dtype=np.float32)
    local_time = np.zeros((batch_size, 1, frames, 5), dtype=np.float32)
    long_time = np.zeros((batch_size, 1, 5), dtype=np.float32)
    objects = np.zeros((batch_size, 1, 1, 12), dtype=np.float32)
    step_mask = np.zeros((batch_size, 1, frames), dtype=bool)
    wavelength_mask = np.zeros((batch_size, 1, frames, 1), dtype=bool)
    for index, example in enumerate(examples):
        count = int(example["raster"].shape[0])  # type: ignore[union-attr]
        raster[index, 0, :count] = example["raster"]  # type: ignore[index]
        wavelength[index, 0, :count] = example["wavelength_tokens"]  # type: ignore[index]
        geometry[index, 0, :count] = example["geometry"]  # type: ignore[index]
        coverage[index, 0, :count] = example["coverage"]  # type: ignore[index]
        local_time[index, 0, :count] = example["local_time"]  # type: ignore[index]
        long_time[index, 0] = example["long_time"]  # type: ignore[index]
        objects[index, 0] = example["object_tokens"]  # type: ignore[index]
        # The last wavelength token is the real adapter's finite-data
        # validity fraction. A cadence with no finite detector pixels is
        # padding from the temporal model's point of view: allowing it into
        # the recurrent/convolutional path creates a false zero-flux step.
        # Finite measurements with non-zero Kepler quality flags remain
        # usable and are retained through the separate coverage channels.
        valid_cadence = (
            np.asarray(example["wavelength_tokens"], dtype=np.float32)[:, 0, 7] > 0.0
        )
        step_mask[index, 0, :count] = valid_cadence
        wavelength_mask[index, 0, :count, 0] = valid_cadence
    # Keep the optimizer's master parameters and input staging tensors in
    # FP32. CUDA autocast below is compute-only.
    floating = torch.float32
    inputs = AstroMambaHInputs(
        raster=torch.from_numpy(raster).to(device=device, dtype=floating),
        wavelength_tokens=torch.from_numpy(wavelength).to(device=device, dtype=floating),
        wavelength_mask=torch.from_numpy(wavelength_mask).to(device=device),
        object_tokens=torch.from_numpy(objects).to(device=device, dtype=floating),
        object_mask=torch.ones((batch_size, 1, 1), dtype=torch.bool, device=device),
        geometry=torch.from_numpy(geometry).to(device=device, dtype=floating),
        exposure_duration=torch.full((batch_size, 1, frames, 1), 1800.0, dtype=floating, device=device),
        coverage_vector=torch.from_numpy(coverage).to(device=device, dtype=floating),
        local_time=torch.from_numpy(local_time).to(device=device, dtype=floating),
        long_time=torch.from_numpy(long_time).to(device=device, dtype=floating),
        visit_mask=torch.ones((batch_size, 1), dtype=torch.bool, device=device),
        step_mask=torch.from_numpy(step_mask).to(device=device),
        source_xy=torch.from_numpy(objects[:, 0, 0, :2]).to(device=device, dtype=torch.float32),
    )
    target = torch.tensor([[float(example["label"])] for example in examples], dtype=torch.float32, device=device)
    return AstroMambaHTrainingBatch(inputs=inputs, target=target)


def _model_config():
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
        # Compact Kepler windows contain far fewer than 256 frames. Process
        # the compact sequence as one spatial chunk to avoid the severe
        # frame-by-frame CPU overhead of the full-frame setting.
        spatial_chunk_size=256,
        decode_heatmaps=False,
    )


def _load_model(checkpoint: Path, device: torch.device) -> AstroMambaHTrainingAdapter:
    model = AstroMambaHTrainingAdapter(_model_config()).to(device, dtype=torch.float32)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    result = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {
        "core.event_logit_scale",
        "core.event_source_weight",
        "core.event_backbone_weight",
        "core.event_photometry_weight",
        "core.temporal_summary_event.0.weight",
        "core.temporal_summary_event.0.bias",
        "core.temporal_summary_event.2.weight",
        "core.temporal_summary_event.2.bias",
        "core.temporal_shape_event.0.weight",
        "core.temporal_shape_event.0.bias",
        "core.temporal_shape_event.2.weight",
        "core.temporal_shape_event.2.bias",
        "core.temporal_robust_event.0.weight",
        "core.temporal_robust_event.0.bias",
        "core.temporal_robust_event.2.weight",
        "core.temporal_robust_event.2.bias",
        "core.temporal_matched_event.0.weight",
        "core.temporal_matched_event.0.bias",
        "core.temporal_matched_event.2.weight",
        "core.temporal_matched_event.2.bias",
        "core.temporal_sequence_event.0.weight",
        "core.temporal_sequence_event.0.bias",
        "core.temporal_sequence_event.2.weight",
        "core.temporal_sequence_event.2.bias",
        "core.temporal_sequence_projection.0.weight",
        "core.temporal_sequence_projection.0.bias",
        "core.temporal_sequence_projection.2.weight",
        "core.temporal_sequence_projection.2.bias",
    }
    unexpected_missing = set(result.missing_keys) - allowed_missing
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing={sorted(unexpected_missing)} unexpected={result.unexpected_keys}")
    return model


def _reset_source_photometry_branch(model: AstroMambaHTrainingAdapter) -> None:
    """Reinitialize only the small branch whose input semantics changed.

    The spatial/temporal backbone remains initialized from the supplied
    checkpoint. This prevents weights trained on the previous feature-6
    constant from biasing the newly aligned robust temporal signal.
    """

    # AstroMambaHTrainingAdapter keeps the actual AstroMambaH module under
    # ``core``; reset only the two source-photometry modules there.
    for module in (model.core.source_photometry_projection, model.core.source_photometry_event):
        module.apply(lambda child: child.reset_parameters() if hasattr(child, "reset_parameters") else None)


def _reset_temporal_event_heads(model: AstroMambaHTrainingAdapter) -> None:
    """Reset compact event heads when their photometry input distribution changes."""

    modules = (
        model.core.temporal_summary_event,
        model.core.temporal_shape_event,
        model.core.temporal_robust_event,
        model.core.temporal_matched_event,
        model.core.temporal_sequence_event,
        model.core.temporal_sequence_projection,
    )
    for module in modules:
        module.apply(lambda child: child.reset_parameters() if hasattr(child, "reset_parameters") else None)
    for module in (
        model.core.temporal_summary_event,
        model.core.temporal_shape_event,
        model.core.temporal_robust_event,
        model.core.temporal_matched_event,
        model.core.temporal_sequence_projection,
    ):
        final = module[-1]
        final.weight.data.zero_()
        final.bias.data.zero_()
    with torch.no_grad():
        model.core.event_logit_scale.fill_(1.0)
        model.core.event_source_weight.fill_(1.0)
        model.core.event_backbone_weight.fill_(0.5)
        model.core.event_photometry_weight.fill_(0.5)


def _freeze_except_temporal_summary(model: AstroMambaHTrainingAdapter) -> None:
    """Train only the compact temporal evidence calibration heads."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (
        model.core.temporal_summary_event,
        model.core.temporal_shape_event,
        model.core.temporal_robust_event,
        model.core.temporal_matched_event,
        model.core.temporal_sequence_event,
        model.core.temporal_sequence_projection,
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


def _freeze_except_event_heads(model: AstroMambaHTrainingAdapter) -> None:
    """Train compact real-domain event heads without changing the backbone."""

    _freeze_except_temporal_summary(model)
    for module in (model.core.source_photometry_projection, model.core.source_photometry_event):
        for parameter in module.parameters():
            parameter.requires_grad = True


def _freeze_except_event_calibration(model: AstroMambaHTrainingAdapter) -> None:
    """Train only the bounded global event calibration parameters."""

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in (
        model.core.event_logit_scale,
        model.core.event_source_weight,
        model.core.event_backbone_weight,
        model.core.event_photometry_weight,
    ):
        parameter.requires_grad = True


def _metrics(
    model,
    root: Path,
    records: list[dict[str, object]],
    device: torch.device,
    batch_size: int,
    *,
    aperture_fraction: float = 1.0,
    raw_tpf_dir: Path | None = None,
    full_tpf_detrend: bool = False,
) -> dict[str, object]:
    model.eval()
    probabilities: list[float] = []
    labels: list[int] = []
    losses: list[float] = []
    with torch.inference_mode():
        for batch_records in _batch_records(records, batch_size, seed=0, shuffle=False):
            batch = _make_batch(
                root,
                batch_records,
                device,
                aperture_fraction=aperture_fraction,
                raw_tpf_dir=raw_tpf_dir,
                full_tpf_detrend=full_tpf_detrend,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
            logits = output["global_event_logits"].float().reshape(-1)
            targets = batch.target.float().reshape(-1)
            losses.extend(
                float(value)
                for value in F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                .cpu()
                .tolist()
            )
            values = logits.sigmoid().cpu().tolist()
            probabilities.extend(float(value) for value in values)
            labels.extend(int(record["label"]) for record in batch_records)
            del batch, output
    predicted = [int(value >= 0.5) for value in probabilities]
    correct = sum(prediction == label for prediction, label in zip(predicted, labels))
    tp = sum(prediction == label == 1 for prediction, label in zip(predicted, labels))
    tn = sum(prediction == label == 0 for prediction, label in zip(predicted, labels))
    fp = sum(prediction == 1 and label == 0 for prediction, label in zip(predicted, labels))
    fn = sum(prediction == 0 and label == 1 for prediction, label in zip(predicted, labels))
    return {
        "samples": len(labels),
        "correct": correct,
        "accuracy": correct / len(labels) if labels else None,
        "mean_bce_loss": sum(losses) / len(losses) if losses else None,
        "mean_probability": sum(probabilities) / len(probabilities) if probabilities else None,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return 0


def _event_loss(output: dict[str, Tensor], batch: AstroMambaHTrainingBatch, smoothing: float) -> Tensor:
    if smoothing <= 0.0:
        return event_only_loss_fn(output, batch)
    target = batch.target.to(dtype=torch.float32)
    logits = output["global_event_logits"]
    softened = target * (1.0 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, softened.reshape_as(logits))


def _mil_event_loss(output: dict[str, Tensor], batch: AstroMambaHTrainingBatch, smoothing: float) -> Tensor:
    """Combine global BCE with sparse multiple-instance temporal supervision."""
    global_loss = _event_loss(output, batch, smoothing)
    frame_logits = output["frame_event_logits"]
    step_mask = batch.inputs.step_mask
    count = step_mask.sum(dim=(1, 2)).clamp_min(1).to(frame_logits.dtype)
    masked = frame_logits.masked_fill(~step_mask, torch.finfo(frame_logits.dtype).min)
    temperature = 0.25
    mil_logits = temperature * (torch.logsumexp(masked / temperature, dim=(1, 2)) - count.log())
    if smoothing > 0.0:
        target = batch.target.to(dtype=torch.float32)
        target = target * (1.0 - smoothing) + 0.5 * smoothing
    else:
        target = batch.target.to(dtype=torch.float32)
    mil_loss = F.binary_cross_entropy_with_logits(mil_logits, target.reshape_as(mil_logits))
    return 0.5 * (global_loss + mil_loss)


def _build_optimizers(
    model: torch.nn.Module,
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[torch.optim.Optimizer], str]:
    if name == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)], "AdamW"
    if name != "muon":
        raise ValueError(f"unsupported optimizer: {name}")
    muon_type = getattr(torch.optim, "Muon", None)
    if muon_type is None:
        raise RuntimeError("optimizer=muon requires a PyTorch build with torch.optim.Muon")
    # Muon is intended for hidden matrix weights. Keep biases, embeddings,
    # normalization parameters, and convolution kernels on AdamW, which is the
    # standard hybrid policy and avoids applying Newton-Schulz updates to
    # parameters for which they are not defined.
    muon_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim == 2]
    adam_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim != 2]
    optimizers: list[torch.optim.Optimizer] = [
        muon_type(muon_parameters, lr=learning_rate, weight_decay=weight_decay, momentum=0.95),
    ]
    if adam_parameters:
        optimizers.append(torch.optim.AdamW(adam_parameters, lr=learning_rate * 0.1, weight_decay=weight_decay))
    return optimizers, "Muon_plus_AdamW"


def _synthetic_helpers():
    """Load the compact synthetic curriculum lazily for real-data rehearsal."""
    from train_synthetic_until_perfect import (  # type: ignore[import-not-found]
        _batch_cache_path,
        _make_batch,
        _synthetic_config,
    )

    return _batch_cache_path, _make_batch, _synthetic_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=64,
        help="batch size used only for train/validation/test metrics",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adamw", "muon"), default="adamw")
    parser.add_argument("--scheduler", choices=("constant", "cosine"), default="constant")
    parser.add_argument("--loss-mode", choices=("event", "mil"), default="event")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--paired-ranking-weight",
        type=float,
        default=0.0,
        help="add a positive-vs-control same-target ranking loss; zero disables it",
    )
    parser.add_argument(
        "--synthetic-rehearsal-pairs",
        type=int,
        default=0,
        help="number of compact synthetic pairs mixed into each real update; zero disables rehearsal",
    )
    parser.add_argument("--synthetic-rehearsal-weight", type=float, default=0.25)
    parser.add_argument("--synthetic-rehearsal-start-index", type=int, default=900000)
    parser.add_argument("--synthetic-cache-dir", type=Path)
    parser.add_argument(
        "--aperture-fraction",
        type=float,
        default=1.0,
        help="fraction of finite median-image pixels used by the direct Kepler photometry path",
    )
    parser.add_argument(
        "--raw-tpf-dir",
        type=Path,
        help="retained Kepler TPF directory used by the optional full-quarter detrending path",
    )
    parser.add_argument(
        "--full-tpf-detrend",
        action="store_true",
        help="detrend local photometry against a degree-3 fit to the retained full TPF quarter",
    )
    parser.add_argument("--seed", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rss-cap-bytes", type=int, default=RSS_CAP_BYTES)
    parser.add_argument(
        "--reset-source-photometry-branch",
        action="store_true",
        help="reinitialize the source-conditioned 2-feature projection/event head after loading the checkpoint",
    )
    parser.add_argument(
        "--reset-temporal-event-heads",
        action="store_true",
        help="reinitialize compact temporal event heads when the photometry input distribution changes",
    )
    parser.add_argument(
        "--train-only-temporal-summary",
        action="store_true",
        help="freeze the representation and train only the compact temporal evidence calibration head",
    )
    parser.add_argument(
        "--train-only-event-calibration",
        action="store_true",
        help="freeze every feature head and train only the bounded global event calibration parameters",
    )
    parser.add_argument(
        "--train-only-event-heads",
        action="store_true",
        help="freeze the spatial/temporal backbone and train compact temporal plus source-photometry event heads",
    )
    parser.add_argument(
        "--include-validation-in-training",
        action="store_true",
        help="fit on train and validation records together after model selection; the test split remains untouched",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.eval_batch_size < 1 or args.epochs < 1 or args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("batch sizes, epochs, and learning-rate must be positive; weight-decay cannot be negative")
    if args.rss_cap_bytes < 1:
        raise ValueError("rss-cap-bytes must be positive")
    if not 0.0 < args.aperture_fraction <= 1.0:
        raise ValueError("aperture-fraction must be in (0, 1]")
    if args.full_tpf_detrend and args.raw_tpf_dir is None:
        raise ValueError("full-tpf-detrend requires --raw-tpf-dir")
    if args.full_tpf_detrend and not args.raw_tpf_dir.is_dir():
        raise FileNotFoundError(args.raw_tpf_dir)
    if args.synthetic_rehearsal_pairs < 0 or args.synthetic_rehearsal_weight < 0.0:
        raise ValueError("synthetic rehearsal pairs and weight cannot be negative")
    if args.paired_ranking_weight < 0.0:
        raise ValueError("paired-ranking-weight cannot be negative")
    if sum(
        int(value)
        for value in (
            args.train_only_temporal_summary,
            args.train_only_event_calibration,
            args.train_only_event_heads,
        )
    ) > 1:
        raise ValueError("event-head training modes are mutually exclusive")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("label-smoothing must be in [0, 1)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda":
        # Bound cuDNN's host-side algorithm-search workspaces under the
        # project RSS cap; TF32 still provides fast CUDA matmuls.
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    root = args.manifest.parent
    records = _records(args.manifest)
    splits = {split: [record for record in records if record.get("split") == split] for split in ("train", "validation", "test")}
    if any(not splits[name] for name in splits):
        raise ValueError(f"manifest split is empty: { {name: len(value) for name, value in splits.items()} }")
    if not args.input_checkpoint.is_file():
        raise FileNotFoundError(args.input_checkpoint)
    training_records = splits["train"] + splits["validation"] if args.include_validation_in_training else splits["train"]
    model = _load_model(args.input_checkpoint, device)
    if args.reset_source_photometry_branch:
        _reset_source_photometry_branch(model)
    if args.reset_temporal_event_heads:
        _reset_temporal_event_heads(model)
    if args.train_only_temporal_summary:
        _freeze_except_temporal_summary(model)
    if args.train_only_event_calibration:
        _freeze_except_event_calibration(model)
    if args.train_only_event_heads:
        _freeze_except_event_heads(model)
    optimizers, optimizer_policy = _build_optimizers(
        model,
        name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.paired_ranking_weight:
        pair_count = len({_pair_key(record) for record in training_records})
        train_steps_per_epoch = (pair_count + max(args.batch_size // 2, 1) - 1) // max(args.batch_size // 2, 1)
    else:
        train_steps_per_epoch = (len(training_records) + args.batch_size - 1) // args.batch_size
    total_steps = max(train_steps_per_epoch * args.epochs, 1)
    schedulers = (
        [torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps) for optimizer in optimizers]
        if args.scheduler == "cosine"
        else []
    )
    loss_fn = _event_loss if args.loss_mode == "event" else _mil_event_loss
    synthetic_batch_cache_path = synthetic_make_batch = synthetic_config_factory = None
    synthetic_config = None
    synthetic_cache_dir = args.synthetic_cache_dir or args.output_checkpoint.parent / "synthetic-rehearsal-cache"
    if args.synthetic_rehearsal_pairs:
        (
            synthetic_batch_cache_path,
            synthetic_make_batch,
            synthetic_config_factory,
        ) = _synthetic_helpers()
        synthetic_config = synthetic_config_factory(seed=23)
        synthetic_cache_dir.mkdir(parents=True, exist_ok=True)
    peak_rss = _rss_bytes()
    eval_batch_size = args.eval_batch_size
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history: list[dict[str, object]] = []
    best_validation_bce = float("inf")
    best_validation_accuracy = float("-inf")
    best_epoch = -1
    best_state_dict: dict[str, Tensor] | None = None
    start_time = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        real_losses: list[float] = []
        rehearsal_losses: list[float] = []
        batch_iterator = (
            _paired_batch_records(training_records, args.batch_size, seed=args.seed + epoch)
            if args.paired_ranking_weight
            else _batch_records(training_records, args.batch_size, seed=args.seed + epoch, shuffle=True)
        )
        for batch_records in batch_iterator:
            batch = _make_batch(
                root,
                batch_records,
                device,
                aperture_fraction=args.aperture_fraction,
                raw_tpf_dir=args.raw_tpf_dir,
                full_tpf_detrend=args.full_tpf_detrend,
            )
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch)
                real_loss = loss_fn(output, batch, args.label_smoothing)
                loss = real_loss
                ranking_loss = None
                if args.paired_ranking_weight:
                    logits = output["global_event_logits"].reshape(-1)
                    positive_logits = logits[0::2]
                    control_logits = logits[1::2]
                    ranking_loss = F.relu(0.25 - positive_logits + control_logits).mean()
                    loss = loss + args.paired_ranking_weight * ranking_loss
                rehearsal_loss = None
                if args.synthetic_rehearsal_pairs:
                    if synthetic_batch_cache_path is None or synthetic_make_batch is None or synthetic_config is None:
                        raise RuntimeError("synthetic rehearsal helpers were not initialized")
                    rehearsal_start = args.synthetic_rehearsal_start_index + global_step * args.synthetic_rehearsal_pairs
                    rehearsal_batch = synthetic_make_batch(
                        synthetic_config,
                        rehearsal_start,
                        args.synthetic_rehearsal_pairs,
                        device,
                        synthetic_batch_cache_path(
                            synthetic_cache_dir,
                            rehearsal_start,
                            args.synthetic_rehearsal_pairs,
                        ),
                    )
                    rehearsal_output = model(rehearsal_batch)
                    rehearsal_loss = event_only_loss_fn(rehearsal_output, rehearsal_batch)
                    loss = real_loss + args.synthetic_rehearsal_weight * rehearsal_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {global_step}: {loss}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for optimizer in optimizers:
                optimizer.step()
            for scheduler in schedulers:
                scheduler.step()
            losses.append(float(loss.detach().float().cpu()))
            real_losses.append(float(real_loss.detach().float().cpu()))
            if rehearsal_loss is not None:
                rehearsal_losses.append(float(rehearsal_loss.detach().float().cpu()))
            global_step += 1
            peak_rss = max(peak_rss, _rss_bytes())
            if peak_rss > args.rss_cap_bytes:
                raise MemoryError(
                    f"RSS cap exceeded at step {global_step}: {peak_rss} > {args.rss_cap_bytes}"
                )
            del batch, output, loss, real_loss
            if ranking_loss is not None:
                del ranking_loss
            if args.synthetic_rehearsal_pairs:
                del rehearsal_batch, rehearsal_output, rehearsal_loss
            if global_step % 10 == 0:
                print(json.dumps({"epoch": epoch, "step": global_step, "loss": losses[-1], "rss_bytes": peak_rss}), flush=True)
        with torch.inference_mode():
            train_metrics = _metrics(
                model,
                root,
                training_records,
                device,
                eval_batch_size,
                aperture_fraction=args.aperture_fraction,
                raw_tpf_dir=args.raw_tpf_dir,
                full_tpf_detrend=args.full_tpf_detrend,
            )
            peak_rss = max(peak_rss, _rss_bytes())
            if peak_rss > args.rss_cap_bytes:
                raise MemoryError(
                    f"RSS cap exceeded during train metrics: {peak_rss} > {args.rss_cap_bytes}"
                )
            validation_metrics = _metrics(
                model,
                root,
                splits["validation"],
                device,
                eval_batch_size,
                aperture_fraction=args.aperture_fraction,
                raw_tpf_dir=args.raw_tpf_dir,
                full_tpf_detrend=args.full_tpf_detrend,
            )
            peak_rss = max(peak_rss, _rss_bytes())
            if peak_rss > args.rss_cap_bytes:
                raise MemoryError(
                    f"RSS cap exceeded during validation metrics: {peak_rss} > {args.rss_cap_bytes}"
                )
        history.append({
            "epoch": epoch,
            "steps": len(losses),
            "mean_loss": sum(losses) / len(losses),
            "mean_real_loss": sum(real_losses) / len(real_losses),
            "mean_rehearsal_loss": sum(rehearsal_losses) / len(rehearsal_losses) if rehearsal_losses else None,
            "train": train_metrics,
            "validation": validation_metrics,
        })
        validation_accuracy = validation_metrics.get("accuracy")
        validation_bce = validation_metrics.get("mean_bce_loss")
        if isinstance(validation_bce, (int, float)) and validation_bce < best_validation_bce:
            best_validation_bce = float(validation_bce)
            if isinstance(validation_accuracy, (int, float)):
                best_validation_accuracy = float(validation_accuracy)
            best_epoch = epoch
            # Keep the selected state on CPU so retaining the best epoch does
            # not add a second device-sized copy under the RSS cap.
            best_state_dict = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    if best_state_dict is None:
        raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state_dict)
    test_metrics = _metrics(
        model,
        root,
        splits["test"],
        device,
        eval_batch_size,
        aperture_fraction=args.aperture_fraction,
        raw_tpf_dir=args.raw_tpf_dir,
        full_tpf_detrend=args.full_tpf_detrend,
    )
    peak_rss = max(peak_rss, _rss_bytes())
    if peak_rss > args.rss_cap_bytes:
        raise MemoryError(f"RSS cap exceeded during test metrics: {peak_rss} > {args.rss_cap_bytes}")
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_checkpoint.with_suffix(args.output_checkpoint.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "base_checkpoint": str(args.input_checkpoint),
            "optimizer_policy": f"{optimizer_policy}_FP32_master_weights",
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
            "amp_enabled": device.type == "cuda",
            "amp_dtype": "bfloat16" if device.type == "cuda" else None,
            "input_mode": "kepler_tpf_compact_32x32_padding_only",
            "training_history": history,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "label_smoothing": args.label_smoothing,
            "loss_mode": args.loss_mode,
            "synthetic_rehearsal_pairs": args.synthetic_rehearsal_pairs,
            "synthetic_rehearsal_weight": args.synthetic_rehearsal_weight,
            "synthetic_rehearsal_start_index": args.synthetic_rehearsal_start_index,
            "weight_decay": args.weight_decay,
            "paired_ranking_weight": args.paired_ranking_weight,
            "best_epoch": best_epoch,
            "best_validation_bce": best_validation_bce,
            "best_validation_accuracy": best_validation_accuracy,
            "selection_metric": "validation_mean_bce_loss",
            "reset_source_photometry_branch": args.reset_source_photometry_branch,
            "reset_temporal_event_heads": args.reset_temporal_event_heads,
            "train_only_temporal_summary": args.train_only_temporal_summary,
            "train_only_event_calibration": args.train_only_event_calibration,
            "train_only_event_heads": args.train_only_event_heads,
            "include_validation_in_training": args.include_validation_in_training,
            "aperture_fraction": args.aperture_fraction,
            "full_tpf_detrend": args.full_tpf_detrend,
            "raw_tpf_dir": str(args.raw_tpf_dir) if args.raw_tpf_dir is not None else None,
        },
        temporary,
    )
    os.replace(temporary, args.output_checkpoint)
    peak_gpu = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    report = {
        "status": "complete",
        "manifest": str(args.manifest),
        "input_checkpoint": str(args.input_checkpoint),
        "output_checkpoint": str(args.output_checkpoint),
        "device": str(device),
        "batch_size": args.batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "optimizer_policy": f"{optimizer_policy}_FP32_master_weights",
        "scheduler": args.scheduler,
        "label_smoothing": args.label_smoothing,
        "loss_mode": args.loss_mode,
        "train_only_temporal_summary": args.train_only_temporal_summary,
        "train_only_event_calibration": args.train_only_event_calibration,
        "train_only_event_heads": args.train_only_event_heads,
        "synthetic_rehearsal_pairs": args.synthetic_rehearsal_pairs,
        "synthetic_rehearsal_weight": args.synthetic_rehearsal_weight,
        "synthetic_rehearsal_start_index": args.synthetic_rehearsal_start_index,
        "weight_decay": args.weight_decay,
        "paired_ranking_weight": args.paired_ranking_weight,
        "global_steps": global_step,
        "dataset_counts": {name: len(value) for name, value in splits.items()},
        "training_sample_count": len(training_records),
        "include_validation_in_training": args.include_validation_in_training,
        "aperture_fraction": args.aperture_fraction,
        "full_tpf_detrend": args.full_tpf_detrend,
        "raw_tpf_dir": str(args.raw_tpf_dir) if args.raw_tpf_dir is not None else None,
        "unique_hosts": {name: len({int(record["kepid"]) for record in value}) for name, value in splits.items()},
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_bce": best_validation_bce,
        "best_validation_accuracy": best_validation_accuracy,
        "selection_metric": "validation_mean_bce_loss",
        "reset_temporal_event_heads": args.reset_temporal_event_heads,
        "test": test_metrics,
        "peak_process_rss_bytes": peak_rss,
        "rss_cap_bytes": args.rss_cap_bytes,
        "rss_within_cap": peak_rss <= args.rss_cap_bytes,
        "peak_gpu_memory_allocated_bytes": peak_gpu,
        "elapsed_seconds": time.time() - start_time,
    }
    report_path = args.output_checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "test": test_metrics, "peak_process_rss_bytes": peak_rss, "peak_gpu_memory_allocated_bytes": peak_gpu, "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))
    gc.collect()
    return 0 if report["rss_within_cap"] and test_metrics["samples"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
