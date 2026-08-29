"""Lazy synthetic batches for bounded AstroMamba-H pretraining smoke runs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import replace

import numpy as np

import torch

from model import AstroMambaHInputs
from synthetic import HubbleSyntheticV2, RealObservationParent, SyntheticConfig, SyntheticGenerator

from .adapters import AstroMambaHTrainingBatch


def iter_synthetic_training_batches(
    config: SyntheticConfig,
    *,
    sample_count: int,
    device: torch.device | str = "cpu",
) -> Iterator[AstroMambaHTrainingBatch]:
    """Generate one model-ready synthetic sample at a time.

    The iterator intentionally retains no dataset-wide cache.  Each sample is
    generated, converted to the real AstroMamba-H input contract, yielded, and
    then becomes eligible for collection when the caller advances the stream.
    The model contract is fixed at 720x1280, so a full-resolution config is
    required here even though the standalone generator supports small arrays.
    Null and injected views alternate deterministically to provide both
    negative and positive pretraining examples without storing a manifest.
    """

    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    if (config.raster_height, config.raster_width) != (720, 1280):
        raise ValueError("synthetic training batches require a 720x1280 raster config")

    target_device = torch.device(device)
    for sample_index in range(sample_count):
        sample_config = replace(config, seed=config.seed + sample_index)
        bundle = SyntheticGenerator(sample_config).generate()
        view_name = "null" if sample_index % 2 else "injected"
        arrays = bundle.as_model_numpy(view_name)
        inputs = AstroMambaHInputs(
            **{
                name: torch.from_numpy(value).to(target_device)
                for name, value in arrays.items()
            }
        )
        view = bundle.null if view_name == "null" else bundle.injected
        target = torch.tensor(
            [[1.0 if view.labels.latent_positive else 0.0]],
            dtype=torch.float32,
            device=target_device,
        )
        yield AstroMambaHTrainingBatch(inputs=inputs, target=target)


def iter_parented_synthetic_training_batches(
    parents: Iterable[RealObservationParent],
    *,
    sample_count: int,
    device: torch.device | str = "cpu",
) -> Iterator[AstroMambaHTrainingBatch]:
    """Lazily inject events into real parents and convert one bundle at a time.

    Parents must already be loaded at the model's 720x1280 raster size.  The
    function intentionally refuses implicit resizing: changing a parent PSF or
    WCS without recording it would invalidate the real-observation contract.
    """

    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    parent_list = tuple(parents)
    if sample_count and not parent_list:
        raise ValueError("at least one parent is required for a non-empty stream")
    target_device = torch.device(device)
    for sample_index in range(sample_count):
        parent = parent_list[sample_index % len(parent_list)]
        if any(
            exposure.science is None
            or exposure.uncertainty is None
            or exposure.dq is None
            or exposure.science.shape != (720, 1280)
            for exposure in parent.exposures
        ):
            raise ValueError("parented training requires every parent image to be 720x1280 with science, uncertainty, and dq")
        result = HubbleSyntheticV2(seed=sample_index).generate(parent, sample_index=sample_index)
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
        raster = np.zeros((1, visits, steps, 6, 720, 1280), dtype=np.float32)
        wavelength_tokens = np.zeros((1, visits, steps, 1, 8), dtype=np.float32)
        wavelength_mask = np.zeros((1, visits, steps, 1), dtype=bool)
        object_tokens = np.zeros((1, visits, 1, 12), dtype=np.float32)
        object_mask = np.ones((1, visits, 1), dtype=bool)
        geometry = np.zeros((1, visits, steps, 10), dtype=np.float32)
        exposure_duration = np.ones((1, visits, steps, 1), dtype=np.float32)
        coverage = np.zeros((1, visits, steps, 6), dtype=np.float32)
        local_time = np.zeros((1, visits, steps, 5), dtype=np.float32)
        long_time = np.zeros((1, visits, 5), dtype=np.float32)
        time_origin = parent.exposures[0].t_mid_bjd_tdb
        for visit_index, group in enumerate(visit_groups):
            visit_midpoints = []
            for step_index, (exposure, injected_exposure) in enumerate(group):
                science = injected_exposure.science
                uncertainty = injected_exposure.uncertainty
                dq = injected_exposure.dq
                scale = max(float(np.nanmedian(uncertainty)), 1.0e-6)
                baseline = float(np.nanmedian(science))
                raster[0, visit_index, step_index, 0] = (science - baseline) / scale
                raster[0, visit_index, step_index, 1] = injected_exposure.relative_flux_drop
                raster[0, visit_index, step_index, 2] = uncertainty / scale
                raster[0, visit_index, step_index, 3] = (dq == 0).astype(np.float32)
                raster[0, visit_index, step_index, 5] = 1.0
                wavelength_tokens[0, visit_index, step_index, 0] = np.array(
                    [
                        np.log10(_filter_wavelength(exposure.filter_name)) / 4.0,
                        float(np.nanmean(science - baseline)) / scale,
                        float(np.nanmean(uncertainty)) / scale,
                        1.0,
                        exposure.exposure_seconds,
                        1.0,
                        float(np.mean(dq == 0)),
                        1.0,
                    ],
                    dtype=np.float32,
                )
                wavelength_mask[0, visit_index, step_index, 0] = True
                observer_position = exposure.observer_position or (0.0, 0.0, 0.0)
                observer_velocity = exposure.observer_velocity or (0.0, 0.0, 0.0)
                roll = float(exposure.pointing.get("roll_deg", 0.0))
                geometry[0, visit_index, step_index] = np.asarray(
                    [
                        parent.source_x / 1280.0,
                        parent.source_y / 720.0,
                        exposure.focus or 0.0,
                        *observer_position,
                        *observer_velocity,
                        roll / 360.0,
                    ],
                    dtype=np.float32,
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
                coverage[0, visit_index, step_index] = np.asarray(
                    [1.0, float(np.mean(dq == 0)), 1.0, 1.0, float(exposure.wcs is not None), float(exposure.observer_position is not None)],
                    dtype=np.float32,
                )
                visit_midpoints.append(exposure.t_mid_bjd_tdb)
            first_exposure = group[0][0]
            observer_position = first_exposure.observer_position or (0.0, 0.0, 0.0)
            observer_velocity = first_exposure.observer_velocity or (0.0, 0.0, 0.0)
            object_tokens[0, visit_index, 0] = np.asarray(
                [
                    parent.source_x / 1280.0,
                    parent.source_y / 720.0,
                    *first_exposure.pixel_scale_arcsec,
                    *observer_position,
                    *observer_velocity,
                    first_exposure.focus or 0.0,
                    1.0,
                ],
                dtype=np.float32,
            )
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
        )
        target = torch.tensor(
            [[float(any(item.relative_flux_drop > 0.0 for item in result.injection.injected))]],
            dtype=torch.float32,
            device=target_device,
        )
        yield AstroMambaHTrainingBatch(inputs=inputs, target=target)


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
