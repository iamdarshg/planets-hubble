"""Lazy synthetic batches for bounded AstroMamba-H pretraining smoke runs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import torch

from model import AstroMambaHInputs
from synthetic import HubbleSyntheticV2, RealObservationParent, SyntheticConfig, SyntheticGenerator

from .adapters import AstroMambaHTrainingBatch
from .cache import (
    DEFAULT_MAX_CACHE_BYTES,
    DEFAULT_MAX_ENTRY_BYTES,
    MAX_CACHE_ENTRIES,
    DiskCachePayload,
    ProceduralSyntheticCache,
)


_PERIOD_CONSTRAINT_STATUS_TO_INDEX = {
    "well_constrained": 0,
    "weakly_constrained": 1,
    "prior_dominated": 2,
    "unconstrained": 3,
}
_PERIODIC_EVENT_TYPES = {"transit", "stellar_spot_modulation", "eclipsing_binary"}


@dataclass
class _CachedView:
    arrays: dict[str, np.ndarray]
    labels: object


@dataclass
class _CachedSyntheticSample:
    """One generated pair, held only for the current iterator step."""

    null: _CachedView
    injected: _CachedView

    @property
    def coverage_vector(self) -> np.ndarray:
        return self.injected.arrays["coverage_vector"][0]

    def as_model_numpy(self, view_name: str) -> dict[str, np.ndarray]:
        if view_name == "null":
            return self.null.arrays
        if view_name == "injected":
            return self.injected.arrays
        raise ValueError("view_name must be null or injected")


def _cache_key(config: SyntheticConfig, sample_index: int) -> str:
    config_json = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    config_digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
    return f"synthetic-v1-{config_digest}-{sample_index}"


def _cache_metadata(view: object, event_mask: np.ndarray) -> dict[str, object]:
    labels = view.labels
    return {
        "latent_positive": bool(labels.latent_positive),
        "event_type": labels.event_type,
        "parameter_constraint_status": labels.parameter_constraint_status,
        "event_mask": event_mask.tolist(),
    }


def _cached_view(
    arrays: dict[str, np.ndarray], metadata: dict[str, object]
) -> _CachedView:
    event_mask = np.asarray(metadata["event_mask"], dtype=np.float32)
    labels = SimpleNamespace(
        latent_positive=bool(metadata["latent_positive"]),
        event_type=metadata.get("event_type"),
        parameter_constraint_status=metadata.get("parameter_constraint_status"),
        event_mask=event_mask,
    )
    return _CachedView(arrays=arrays, labels=labels)


def _sample_from_bundle(bundle: object, config: SyntheticConfig) -> _CachedSyntheticSample:
    null_mask = _event_mask(bundle.null.labels, config.visits, config.local_steps)
    injected_mask = _event_mask(bundle.injected.labels, config.visits, config.local_steps)
    return _CachedSyntheticSample(
        null=_cached_view(
            bundle.as_model_numpy("null"), _cache_metadata(bundle.null, null_mask)
        ),
        injected=_cached_view(
            bundle.as_model_numpy("injected"),
            _cache_metadata(bundle.injected, injected_mask),
        ),
    )


def _payload_from_sample(
    sample: _CachedSyntheticSample,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays: dict[str, np.ndarray] = {}
    for view_name, view in (("null", sample.null), ("injected", sample.injected)):
        arrays.update({f"{view_name}__{name}": value for name, value in view.arrays.items()})
    visits, steps = sample.coverage_vector.shape[:2]
    metadata = {
        "null": _cache_metadata(
            sample.null,
            _event_mask(sample.null.labels, visits, steps),
        ),
        "injected": _cache_metadata(
            sample.injected,
            _event_mask(sample.injected.labels, visits, steps),
        ),
    }
    return arrays, metadata


def _sample_from_payload(payload: DiskCachePayload) -> _CachedSyntheticSample:
    metadata = payload.metadata
    if not isinstance(metadata.get("null"), dict) or not isinstance(metadata.get("injected"), dict):
        raise ValueError("synthetic cache metadata is incomplete")
    null_arrays = {
        name.removeprefix("null__"): value
        for name, value in payload.arrays.items()
        if name.startswith("null__")
    }
    injected_arrays = {
        name.removeprefix("injected__"): value
        for name, value in payload.arrays.items()
        if name.startswith("injected__")
    }
    if not null_arrays or not injected_arrays:
        raise ValueError("synthetic cache entry has no model arrays")
    return _CachedSyntheticSample(
        null=_cached_view(null_arrays, metadata["null"]),
        injected=_cached_view(injected_arrays, metadata["injected"]),
    )


def _load_or_generate_sample(
    config: SyntheticConfig,
    sample_index: int,
    cache: ProceduralSyntheticCache | None,
) -> _CachedSyntheticSample:
    key = _cache_key(config, sample_index)
    if cache is not None:
        payload = cache.load(key)
        if payload is not None:
            try:
                return _sample_from_payload(payload)
            except (KeyError, TypeError, ValueError):
                pass
    bundle = SyntheticGenerator(replace(config, seed=config.seed + sample_index)).generate()
    sample = _sample_from_bundle(bundle, config)
    if cache is not None:
        arrays, metadata = _payload_from_sample(sample)
        cache.store(key, arrays, metadata)
    return sample


def _open_cache(
    cache_dir: str | Path | None,
    *,
    cache_size_mib: int | None,
    max_cache_entries: int,
    max_entry_bytes: int,
) -> ProceduralSyntheticCache | None:
    if cache_dir is None:
        return None
    if cache_size_mib is not None and cache_size_mib < 1:
        raise ValueError("cache_size must be a positive MiB budget")
    max_cache_bytes = (
        DEFAULT_MAX_CACHE_BYTES
        if cache_size_mib is None
        else cache_size_mib * 1024 * 1024
    )
    return ProceduralSyntheticCache(
        cache_dir,
        max_cache_entries=max_cache_entries,
        max_cache_bytes=max_cache_bytes,
        max_entry_bytes=max_entry_bytes,
    )


def _event_mask(labels: object, visits: int, steps: int) -> np.ndarray:
    value = getattr(labels, "event_mask", None)
    if value is None:
        return np.zeros((visits, steps), dtype=np.float32)
    mask = np.asarray(value, dtype=np.float32)
    if mask.shape != (visits, steps):
        raise ValueError("synthetic event_mask shape does not match the configured sequence")
    return mask


def _period_constraint_target(labels: object) -> int | None:
    event_type = getattr(labels, "event_type", None)
    if not getattr(labels, "latent_positive", False) or event_type not in _PERIODIC_EVENT_TYPES:
        return None
    status = getattr(labels, "parameter_constraint_status", None)
    if status == "unconstrained":
        return None
    return _PERIOD_CONSTRAINT_STATUS_TO_INDEX.get(status)


def _synthetic_auxiliary_targets(
    bundle: object,
    view: object,
    *,
    source_x: float,
    source_y: float,
    target_device: torch.device,
    source_top_k: int,
) -> dict[str, torch.Tensor]:
    visits, steps = bundle.coverage_vector.shape[:2]
    event_mask = _event_mask(view.labels, visits, steps)
    source_target = _source_target_map(visits, steps, source_x, source_y)
    candidate_target = source_target * event_mask[..., None, None]
    result = {
        "candidate": torch.tensor(
            [float(view.labels.latent_positive)], dtype=torch.float32, device=target_device
        ),
        "candidate_heatmap": torch.from_numpy(candidate_target[None]).to(target_device),
        "source": torch.from_numpy(source_target[None]).to(target_device),
        "artifact": None,
        "ood": None,
        "coverage": torch.tensor(
            [float(np.mean(bundle.coverage_vector[..., 0]))],
            dtype=torch.float32,
            device=target_device,
        ),
        "sufficiency": torch.tensor(
            [float(np.mean(bundle.coverage_vector[..., 2]))],
            dtype=torch.float32,
            device=target_device,
        ),
        "visit_event": torch.from_numpy(event_mask.any(axis=1).astype(np.float32))[None].to(target_device),
        "frame_event": torch.from_numpy(event_mask)[None].to(target_device),
        "source_event": torch.from_numpy(
            _source_event_target(bool(view.labels.latent_positive), source_top_k)
        )[None].to(target_device),
    }
    period_target = _period_constraint_target(view.labels)
    if period_target is not None:
        result["period_constraint"] = torch.tensor(
            [period_target], dtype=torch.long, device=target_device
        )
    return {name: value for name, value in result.items() if value is not None}


def iter_synthetic_training_batches(
    config: SyntheticConfig,
    *,
    sample_count: int,
    device: torch.device | str = "cpu",
    source_top_k: int = 96,
    cache_dir: str | Path | None = None,
    cache_size: int | None = None,
    max_cache_entries: int = MAX_CACHE_ENTRIES,
    max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
) -> Iterator[AstroMambaHTrainingBatch]:
    """Generate one model-ready synthetic sample at a time.

    Generation is procedural: each sample is created immediately before it is
    yielded (or loaded from the SSD cache when the same descriptor already
    exists).  The optional cache_dir names an explicit SSD-backed directory;
    the cache stores at most max_cache_entries (64 by default) compressed NPZ
    pairs and never retains full arrays in process memory.  cache_size is an
    optional total budget in MiB.  With cache_dir=None no cache is used.
    Null and injected views alternate deterministically to provide both
    negative and positive pretraining examples without storing a manifest.
    """

    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    if (config.raster_height, config.raster_width) != (720, 1280):
        raise ValueError("synthetic training batches require a 720x1280 raster config")
    if source_top_k < 1:
        raise ValueError("source_top_k must be positive")

    target_device = torch.device(device)
    cache = _open_cache(
        cache_dir,
        cache_size_mib=cache_size,
        max_cache_entries=max_cache_entries,
        max_entry_bytes=max_entry_bytes,
    )
    for sample_index in range(sample_count):
        bundle = _load_or_generate_sample(config, sample_index, cache)
        view_name = "null" if sample_index % 2 else "injected"
        arrays = bundle.as_model_numpy(view_name)
        inputs = AstroMambaHInputs(
            **{
                name: torch.from_numpy(value).to(target_device)
                for name, value in arrays.items()
            },
            source_xy=torch.tensor(
                [[config.source_x, config.source_y]], dtype=torch.float32, device=target_device
            ),
        )
        view = bundle.null if view_name == "null" else bundle.injected
        target = torch.tensor(
            [[1.0 if view.labels.latent_positive else 0.0]],
            dtype=torch.float32,
            device=target_device,
        )
        auxiliary_targets = _synthetic_auxiliary_targets(
            bundle,
            view,
            source_x=config.source_x,
            source_y=config.source_y,
            target_device=target_device,
            source_top_k=source_top_k,
        )
        yield AstroMambaHTrainingBatch(
            inputs=inputs, target=target, auxiliary_targets=auxiliary_targets
        )


def iter_paired_synthetic_training_batches(
    config: SyntheticConfig,
    *,
    sample_count: int,
    device: torch.device | str = "cpu",
    source_top_k: int = 96,
    cache_dir: str | Path | None = None,
    cache_size: int | None = None,
    max_cache_entries: int = MAX_CACHE_ENTRIES,
    max_entry_bytes: int = DEFAULT_MAX_ENTRY_BYTES,
    start_index: int = 0,
) -> Iterator[AstroMambaHTrainingBatch]:
    """Yield null and injected counterfactual views from one shared scene.

    The pair is generated once and stacked along the batch axis, preserving
    the common noise/nuisance realization for contrastive or ranking losses.
    cache_dir enables the same bounded SSD-backed procedural cache used by
    iter_synthetic_training_batches; cache_size is a total MiB budget.
    start_index offsets the procedural sample counter so a resumed run can
    continue from exactly where the previous process stopped.
    """

    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    if (config.raster_height, config.raster_width) != (720, 1280):
        raise ValueError("synthetic training batches require a 720x1280 raster config")
    if source_top_k < 1:
        raise ValueError("source_top_k must be positive")
    target_device = torch.device(device)
    cache = _open_cache(
        cache_dir,
        cache_size_mib=cache_size,
        max_cache_entries=max_cache_entries,
        max_entry_bytes=max_entry_bytes,
    )
    for local_index in range(sample_count):
        sample_index = start_index + local_index
        bundle = _load_or_generate_sample(config, sample_index, cache)
        views = (bundle.null, bundle.injected)
        numpy_views = [bundle.as_model_numpy(name) for name in ("null", "injected")]
        inputs = AstroMambaHInputs(
            **{
                name: torch.cat(
                    [torch.from_numpy(view[name]) for view in numpy_views], dim=0
                ).to(target_device)
                for name in numpy_views[0]
            },
            source_xy=torch.tensor(
                [[config.source_x, config.source_y], [config.source_x, config.source_y]],
                dtype=torch.float32,
                device=target_device,
            ),
        )
        targets = torch.tensor(
            [[float(view.labels.latent_positive)] for view in views],
            dtype=torch.float32,
            device=target_device,
        )
        frame_targets = []
        visit_targets = []
        candidate_targets = []
        source_targets = []
        coverage_targets = []
        sufficiency_targets = []
        for view in views:
            mask = _event_mask(view.labels, config.visits, config.local_steps)
            source_target = _source_target_map(
                config.visits, config.local_steps, config.source_x, config.source_y
            )
            frame_targets.append(torch.from_numpy(mask))
            visit_targets.append(torch.from_numpy(mask.any(axis=1).astype(np.float32)))
            source_targets.append(torch.from_numpy(source_target))
            candidate_targets.append(torch.from_numpy(source_target * mask[..., None, None]))
            coverage_targets.append(float(np.mean(bundle.coverage_vector[..., 0])))
            sufficiency_targets.append(float(np.mean(bundle.coverage_vector[..., 2])))
        period_targets = [
            _period_constraint_target(view.labels) for view in views
        ]
        auxiliary_targets = {
            "candidate": targets[:, 0],
            "candidate_heatmap": torch.stack(candidate_targets).to(target_device),
            "source": torch.stack(source_targets).to(target_device),
            "coverage": torch.tensor(coverage_targets, dtype=torch.float32, device=target_device),
            "sufficiency": torch.tensor(
                sufficiency_targets, dtype=torch.float32, device=target_device
            ),
            "frame_event": torch.stack(frame_targets).to(target_device),
            "visit_event": torch.stack(visit_targets).to(target_device),
            "source_event": torch.from_numpy(
                np.stack(
                    [
                        _source_event_target(bool(view.labels.latent_positive), source_top_k)
                        for view in views
                    ]
                )
            ).to(target_device),
        }
        valid_period_targets = [value for value in period_targets if value is not None]
        if len(valid_period_targets) == len(period_targets):
            auxiliary_targets["period_constraint"] = torch.tensor(
                valid_period_targets, dtype=torch.long, device=target_device
            )
        yield AstroMambaHTrainingBatch(
            inputs=inputs,
            target=targets,
            auxiliary_targets=auxiliary_targets,
        )


def iter_parented_synthetic_training_batches(
    parents: Iterable[RealObservationParent],
    *,
    sample_count: int,
    device: torch.device | str = "cpu",
    start_index: int = 0,
    sequence_summary: bool = False,
    source_top_k: int = 96,
) -> Iterator[AstroMambaHTrainingBatch]:
    """Lazily inject events into real parents and convert one bundle at a time.

    Parents must already be loaded at the model's 720x1280 raster size.  The
    function intentionally refuses implicit resizing: changing a parent PSF or
    WCS without recording it would invalidate the real-observation contract.
    ``sequence_summary`` consumes every frame and reduces the parent to one
    temporal spatial summary for the GPU path.  It is a cap-safe fallback, not
    a replacement for full local/long-time sequence processing.
    """

    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    if source_top_k < 1:
        raise ValueError("source_top_k must be positive")
    parent_list = tuple(parents)
    if sample_count and not parent_list:
        raise ValueError("at least one parent is required for a non-empty stream")
    target_device = torch.device(device)
    for local_index in range(sample_count):
        sample_index = start_index + local_index
        parent = parent_list[sample_index % len(parent_list)]
        if any(
            exposure.science is None
            or exposure.uncertainty is None
            or exposure.dq is None
            or exposure.science.shape != (720, 1280)
            for exposure in parent.exposures
        ):
            raise ValueError("parented training requires every parent image to be 720x1280 with science, uncertainty, and dq")
        requested_event = "planet_transit" if sample_index % 2 == 0 else "null"
        result = HubbleSyntheticV2(seed=sample_index).generate(
            parent,
            sample_index=sample_index,
            event_type=requested_event,
        )
        selected = result.injection.injected if sample_index % 2 == 0 else result.injection.null
        grouped: dict[str, list[tuple[object, object]]] = {}
        by_id = {exposure.exposure_id: exposure for exposure in parent.exposures}
        for injected_exposure in selected:
            grouped.setdefault(by_id[injected_exposure.exposure_id].visit_id, []).append(
                (by_id[injected_exposure.exposure_id], injected_exposure)
            )
        visit_groups = list(grouped.values())
        steps = max(len(group) for group in visit_groups)
        visits = len(visit_groups)
        # The calibrated parent pixels are already normalized and clipped
        # below.  FP16 storage avoids a second 720p sequence-sized copy while
        # the GPU worker converts/consumes the batch under autocast.
        raster = np.zeros((1, visits, steps, 6, 720, 1280), dtype=np.float16)
        wavelength_tokens = np.zeros((1, visits, steps, 1, 8), dtype=np.float16)
        wavelength_mask = np.zeros((1, visits, steps, 1), dtype=bool)
        parent_objects = parent.object_tokens
        object_count = 1 if parent_objects is None else parent_objects.shape[0]
        if parent_objects is not None and parent_objects.shape[1] != 12:
            raise ValueError("parent object_tokens must have 12 features for AstroMamba-H")
        object_tokens = np.zeros((1, visits, object_count, 12), dtype=np.float16)
        object_mask = np.ones((1, visits, object_count), dtype=bool)
        if parent.object_mask is not None:
            if parent.object_mask.shape[0] != object_count:
                raise ValueError("parent object_mask does not match parent object_tokens")
            object_mask[:] = parent.object_mask[None, None, :]
        geometry = np.zeros((1, visits, steps, 10), dtype=np.float16)
        exposure_duration = np.ones((1, visits, steps, 1), dtype=np.float16)
        coverage = np.zeros((1, visits, steps, 6), dtype=np.float16)
        local_time = np.zeros((1, visits, steps, 5), dtype=np.float32)
        long_time = np.zeros((1, visits, 5), dtype=np.float32)
        step_valid = np.zeros((1, visits, steps), dtype=bool)
        frame_targets = np.zeros((visits, steps), dtype=np.float32)
        time_origin = parent.exposures[0].t_mid_bjd_tdb
        for visit_index, group in enumerate(visit_groups):
            visit_midpoints = []
            for step_index, (exposure, injected_exposure) in enumerate(group):
                science = injected_exposure.science
                uncertainty = injected_exposure.uncertainty
                dq = injected_exposure.dq
                science = np.nan_to_num(science, nan=0.0, posinf=0.0, neginf=0.0)
                uncertainty = np.nan_to_num(np.abs(uncertainty), nan=0.0, posinf=0.0, neginf=0.0)
                baseline = float(np.nanmedian(science))
                residual = science - baseline
                robust_scale = float(np.nanpercentile(np.abs(residual), 75.0))
                uncertainty_scale = float(np.nanpercentile(uncertainty, 75.0))
                scale = max(robust_scale, uncertainty_scale, 1.0)
                # Both signal channels must be observable from the parent
                # exposure.  Do not write the known synthetic label/drop into
                # the raster: that would make training trivially leak the
                # target and would be unavailable at inference time.
                raster[0, visit_index, step_index, 0] = np.clip(
                    science / max(abs(baseline), 1.0) - 1.0, -20.0, 20.0
                )
                raster[0, visit_index, step_index, 1] = np.clip(residual / scale, -20.0, 20.0)
                raster[0, visit_index, step_index, 2] = np.clip(uncertainty / scale, 0.0, 20.0)
                raster[0, visit_index, step_index, 3] = (dq == 0).astype(np.float32)
                raster[0, visit_index, step_index, 5] = 1.0
                step_valid[0, visit_index, step_index] = True
                frame_targets[visit_index, step_index] = injected_exposure.relative_flux_drop > 0.0
                wavelength_tokens[0, visit_index, step_index, 0] = np.array(
                    [
                        np.log10(_filter_wavelength(exposure.filter_name)) / 4.0,
                        _source_aperture_residual(
                            residual, scale, parent.source_x, parent.source_y
                        ),
                        _source_aperture_uncertainty(
                            uncertainty, scale, parent.source_x, parent.source_y
                        ),
                        1.0,
                        exposure.exposure_seconds,
                        1.0,
                        float(np.mean(dq == 0)),
                        1.0,
                    ],
                    dtype=np.float16,
                )
                wavelength_mask[0, visit_index, step_index, 0] = True
                observer_position = exposure.observer_position or (0.0, 0.0, 0.0)
                observer_velocity = exposure.observer_velocity or (0.0, 0.0, 0.0)
                roll = float(exposure.pointing.get("roll_deg", 0.0))
                geometry[0, visit_index, step_index] = np.asarray(
                    [
                        *observer_position,
                        *observer_velocity,
                        roll / 360.0,
                        float(exposure.pointing.get("off_axis_angle_deg", 0.0)) / 180.0,
                        float(exposure.pointing.get("boresight_ra_deg", 0.0)) / 360.0,
                        float(exposure.pointing.get("boresight_dec_deg", 0.0)) / 90.0,
                    ],
                    dtype=np.float16,
                )
                exposure_duration[0, visit_index, step_index, 0] = exposure.exposure_seconds
                local_time[0, visit_index, step_index] = np.asarray(
                    [
                        exposure.t_start_bjd_tdb - time_origin,
                        exposure.t_mid_bjd_tdb - time_origin,
                        exposure.t_end_bjd_tdb - time_origin,
                        exposure.exposure_seconds / 86400.0,
                        1.0,
                    ],
                    dtype=np.float32,
                )
                angular_size = exposure.angular_size_arcsec or (1.0, 1.0)
                coverage[0, visit_index, step_index] = np.asarray(
                    [
                        1.0,
                        float(np.mean(dq == 0)),
                        1.0,
                        angular_size[0] / 100.0,
                        angular_size[1] / 100.0,
                        float(exposure.observer_position is not None),
                    ],
                    dtype=np.float32,
                )
                visit_midpoints.append(exposure.t_mid_bjd_tdb)
            first_exposure = group[0][0]
            observer_position = first_exposure.observer_position or (0.0, 0.0, 0.0)
            observer_velocity = first_exposure.observer_velocity or (0.0, 0.0, 0.0)
            if parent_objects is None:
                angular_size = first_exposure.angular_size_arcsec or (1.0, 1.0)
                object_tokens[0, visit_index, 0] = np.asarray(
                    [
                        parent.source_x / 1280.0,
                        parent.source_y / 720.0,
                        *first_exposure.pixel_scale_arcsec,
                        *observer_position,
                        *observer_velocity,
                        first_exposure.focus or 0.0,
                        float(np.mean(angular_size)) / 100.0,
                    ],
                    dtype=np.float32,
                )
            else:
                object_tokens[0, visit_index] = parent_objects
            long_time[0, visit_index] = np.asarray(
                [
                    min(visit_midpoints) - time_origin,
                    max(visit_midpoints) - time_origin,
                    len(group),
                    max(visit_midpoints) - min(visit_midpoints),
                    1.0,
                ],
                dtype=np.float32,
            )
        # Quality labels are derived from observed validity/DQ coverage. The
        # parent injector has no artifact or OOD truth, so those heads remain
        # intentionally unsupervised for this stream.
        coverage_target = float(np.mean(coverage[..., 1]))
        sufficiency_target = float(np.mean(coverage[..., 1] * coverage[..., 2]))
        if sequence_summary:
            valid_indices = np.flatnonzero(step_valid[0].reshape(-1))
            frame_values = raster[0].reshape(-1, 6, 720, 1280)[valid_indices]
            summary = np.zeros((6, 720, 1280), dtype=np.float16)
            summary[0] = np.median(frame_values[:, 0].astype(np.float32), axis=0).astype(np.float16)
            summary[1] = np.min(frame_values[:, 1].astype(np.float32), axis=0).astype(np.float16)
            summary[2] = np.median(frame_values[:, 2].astype(np.float32), axis=0).astype(np.float16)
            summary[3] = np.max(frame_values[:, 3].astype(np.float32), axis=0).astype(np.float16)
            summary[4] = np.max(frame_values[:, 4].astype(np.float32), axis=0).astype(np.float16)
            summary[5] = np.mean(frame_values[:, 5].astype(np.float32), axis=0).astype(np.float16)
            raster = summary[None, None, None]

            wavelength_values = wavelength_tokens[0].reshape(-1, 1, 8)[valid_indices]
            wavelength_tokens = np.median(
                wavelength_values.astype(np.float32), axis=0
            )[None, None, None].astype(np.float16)
            wavelength_mask = np.ones((1, 1, 1, 1), dtype=bool)
            geometry_values = geometry[0].reshape(-1, 10)[valid_indices]
            geometry = np.mean(geometry_values.astype(np.float32), axis=0)[None, None, None].astype(np.float16)
            exposure_duration = np.array(
                [[[float(exposure_duration[0].reshape(-1)[valid_indices].sum())]]],
                dtype=np.float16,
            )[..., None]
            coverage_values = coverage[0].reshape(-1, 6)[valid_indices]
            coverage = np.mean(coverage_values.astype(np.float32), axis=0)[None, None, None].astype(np.float16)
            local_values = local_time[0].reshape(-1, 5)[valid_indices]
            local_time = np.array(
                [[[[
                    float(local_values[:, 0].min()),
                    float(local_values[:, 1].mean()),
                    float(local_values[:, 2].max()),
                    float(local_values[:, 3].sum()),
                    1.0,
                ]]]],
                dtype=np.float32,
            )
            long_time = np.array(
                [[[
                    float(local_values[:, 0].min()),
                    float(local_values[:, 2].max()),
                    float(len(valid_indices)),
                    float(local_values[:, 2].max() - local_values[:, 0].min()),
                    1.0,
                ]]],
                dtype=np.float32,
            )
            object_tokens = object_tokens.mean(axis=1, keepdims=True)
            object_mask = np.any(object_mask, axis=1, keepdims=True)
            step_valid = np.ones((1, 1, 1), dtype=bool)
            frame_targets = np.array([[bool(frame_targets.any())]], dtype=np.float32)
            visits = steps = 1
        inputs = AstroMambaHInputs(
            raster=torch.from_numpy(raster).to(target_device),
            wavelength_tokens=torch.from_numpy(wavelength_tokens).to(target_device),
            wavelength_mask=torch.from_numpy(wavelength_mask).to(target_device),
            object_tokens=torch.from_numpy(object_tokens).to(target_device),
            object_mask=torch.from_numpy(object_mask).to(target_device),
            geometry=torch.from_numpy(geometry).to(target_device),
            exposure_duration=torch.from_numpy(exposure_duration).to(target_device),
            coverage_vector=torch.from_numpy(coverage).to(target_device),
            local_time=torch.from_numpy(local_time).to(target_device),
            long_time=torch.from_numpy(long_time).to(target_device),
            visit_mask=torch.ones((1, visits), dtype=torch.bool, device=target_device),
            step_mask=torch.from_numpy(step_valid).to(target_device),
            source_xy=torch.tensor(
                [[parent.source_x / 1280.0, parent.source_y / 720.0]],
                dtype=torch.float32,
                device=target_device,
            ),
        )
        target = torch.tensor(
            [[
                float(
                    sample_index % 2 == 0
                    and any(item.relative_flux_drop > 0.0 for item in selected)
                )
            ]],
            dtype=torch.float32,
            device=target_device,
        )
        source_target = _source_target_map(
            visits, steps, parent.source_x / 1280.0, parent.source_y / 720.0
        )
        candidate_target = source_target * frame_targets[..., None, None]
        yield AstroMambaHTrainingBatch(
            inputs=inputs,
            target=target,
            auxiliary_targets={
                "candidate": target[:, 0],
                "candidate_heatmap": torch.from_numpy(candidate_target[None]).to(target_device),
                "source": torch.from_numpy(source_target[None]).to(target_device),
                "coverage": torch.tensor(
                    [coverage_target], dtype=torch.float32, device=target_device
                ),
                "sufficiency": torch.tensor(
                    [sufficiency_target], dtype=torch.float32, device=target_device
                ),
                "visit_event": torch.from_numpy(frame_targets.any(axis=1)[None]).to(target_device),
                "frame_event": torch.from_numpy(frame_targets[None]).to(target_device),
                "source_event": torch.from_numpy(
                    _source_event_target(
                        bool(any(item.relative_flux_drop > 0.0 for item in selected)),
                        source_top_k,
                    )
                )[None].to(target_device),
            },
        )


def _filter_wavelength(filter_name: str) -> float:
    return {
        "F275W": 270.0,
        "F336W": 335.0,
        "F438W": 432.0,
        "F606W": 590.0,
        "F814W": 800.0,
        "F105W": 1050.0,
        "F125W": 1250.0,
        "F140W": 1400.0,
        "F160W": 1540.0,
    }.get(filter_name, 600.0)


def _source_window(
    array: np.ndarray, source_x: float, source_y: float, radius: int = 5
) -> np.ndarray:
    """Return a bounded aperture around the known/catalogued source anchor."""

    height, width = array.shape[-2:]
    center_x = int(round(source_x))
    center_y = int(round(source_y))
    x0 = max(center_x - radius, 0)
    x1 = min(center_x + radius + 1, width)
    y0 = max(center_y - radius, 0)
    y1 = min(center_y + radius + 1, height)
    return array[y0:y1, x0:x1]


def _source_aperture_residual(
    residual: np.ndarray, scale: float, source_x: float, source_y: float
) -> float:
    value = float(np.nanmean(_source_window(residual, source_x, source_y)) / scale)
    return float(np.clip(value, -20.0, 20.0))


def _source_aperture_uncertainty(
    uncertainty: np.ndarray, scale: float, source_x: float, source_y: float
) -> float:
    value = float(np.nanmean(_source_window(uncertainty, source_x, source_y)) / scale)
    return float(np.clip(value, 0.0, 20.0))


def _source_target_map(visits: int, steps: int, x: float, y: float) -> np.ndarray:
    """Create a small supervised source-proposal target on the /8 grid."""
    height, width = 90, 160
    yy, xx = np.mgrid[:height, :width]
    center_x = x * (width - 1)
    center_y = y * (height - 1)
    target = np.exp(-0.5 * (((xx - center_x) / 1.5) ** 2 + ((yy - center_y) / 1.5) ** 2))
    return np.broadcast_to(target.astype(np.float32), (visits, steps, height, width)).copy()


def _source_event_target(positive: bool, source_count: int) -> np.ndarray:
    """Label the known source anchor while keeping extra proposals negative."""

    target = np.zeros(source_count, dtype=np.float32)
    target[0] = float(positive)
    return target
