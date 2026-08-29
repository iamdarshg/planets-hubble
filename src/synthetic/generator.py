"""Deterministic, bounded synthetic observation generation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .models import EventLabels, ObservationView, SyntheticBundle, SyntheticConfig


class SyntheticGenerator:
    """Generate paired null/injected observation bundles with NumPy only.

    The generator models an unresolved source. Its transit path is a uniform
    stellar disk and circular occultor using the analytic two-circle overlap
    area. This is a validated baseline for the simulator, not a full
    limb-darkened stellar surface or planetary microlensing solver.
    """

    RASTER_CHANNELS = (
        "physical_ratio_flux",
        "noise_scaled_residual",
        "normalized_uncertainty",
        "validity_mask",
        "interpolation_mask",
        "exposure_coverage_fraction",
    )
    WAVELENGTH_FEATURES = (
        "normalized_log_wavelength",
        "physical_ratio_flux",
        "noise_scaled_residual",
        "normalized_uncertainty",
        "normalized_bandwidth",
        "normalized_log_exposure",
        "response_integral",
        "validity_mask",
    )

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()

    def generate(self) -> SyntheticBundle:
        config = self.config
        rng = np.random.default_rng(config.seed)
        starts, mids, ends = self._schedule(rng)
        exposure_days = config.exposure_seconds / 86400.0
        wavelengths = np.asarray(config.wavelength_nm, dtype=np.float64)
        wavelength_coordinate = self._normalize_log_wavelength(wavelengths)
        valid_exposure, interpolation, wavelength_mask = self._masks(rng)

        latent_null, latent_injected, event_mask, depth = self._latent_signals(
            mids, starts, ends, wavelengths
        )
        noisy_null, noisy_injected, uncertainty = self._apply_noise(
            rng, latent_null, latent_injected, exposure_days
        )
        pixel_noise = rng.normal(
            0.0,
            0.0002,
            size=(
                config.visits,
                config.local_steps,
                config.raster_height,
                config.raster_width,
            ),
        )

        null_raster = self._render_raster(
            noisy_null,
            uncertainty,
            valid_exposure,
            interpolation,
            wavelength_mask,
            pixel_noise,
        )
        injected_raster = self._render_raster(
            noisy_injected,
            uncertainty,
            valid_exposure,
            interpolation,
            wavelength_mask,
            pixel_noise,
        )
        null_tokens = self._wavelength_tokens(
            noisy_null, uncertainty, wavelength_coordinate, valid_exposure, interpolation, wavelength_mask
        )
        injected_tokens = self._wavelength_tokens(
            noisy_injected,
            uncertainty,
            wavelength_coordinate,
            valid_exposure,
            interpolation,
            wavelength_mask,
        )

        null_labels = self._labels(
            event_mask=None,
            depth=np.zeros_like(depth),
            latent_positive=False,
            injection_seed=None,
            mids=mids,
        )
        injected_labels = self._labels(
            event_mask=event_mask,
            depth=depth,
            latent_positive=bool(event_mask.any()),
            injection_seed=config.seed,
            mids=mids,
        )
        objects, object_mask, object_metadata = self._objects()
        geometry, coverage, local_time, long_time = self._context(
            starts, mids, ends, valid_exposure, interpolation, wavelength_mask
        )
        source_metadata = {
            "source_id": "synthetic-host-0001",
            "ra_deg": 180.0,
            "dec_deg": 20.0,
            "pixel_x": config.source_x * (config.raster_width - 1),
            "pixel_y": config.source_y * (config.raster_height - 1),
            "normalization": "robust_baseline_reference_1.0",
        }
        return SyntheticBundle(
            null=ObservationView(null_raster, null_tokens, wavelength_mask.copy(), null_labels),
            injected=ObservationView(
                injected_raster, injected_tokens, wavelength_mask.copy(), injected_labels
            ),
            timestamps_start_bjd_tdb=starts,
            timestamps_mid_bjd_tdb=mids,
            timestamps_end_bjd_tdb=ends,
            exposure_duration_seconds=np.full(mids.shape, config.exposure_seconds, dtype=np.float32),
            wavelength_nm=wavelengths.astype(np.float32),
            object_tokens=objects,
            object_mask=object_mask,
            geometry=geometry,
            coverage_vector=coverage,
            local_time=local_time,
            long_time=long_time,
            source_metadata=source_metadata,
            object_metadata=object_metadata,
            labels=injected_labels,
        )

    def _schedule(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        config = self.config
        mids = np.empty((config.visits, config.local_steps), dtype=np.float64)
        for visit in range(config.visits):
            base = config.start_bjd_tdb + visit * config.visit_spacing_days
            offsets = np.arange(config.local_steps, dtype=np.float64) * config.local_step_spacing_days
            if config.timestamp_jitter_days:
                offsets += rng.uniform(
                    -config.timestamp_jitter_days,
                    config.timestamp_jitter_days,
                    size=config.local_steps,
                )
            mids[visit] = base + np.sort(offsets)
        half = config.exposure_seconds / 172800.0
        return mids - half, mids, mids + half

    def _masks(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        config = self.config
        valid = np.ones((config.visits, config.local_steps), dtype=np.float32)
        interpolation = np.zeros_like(valid)
        for visit, step in config.invalid_exposures:
            self._check_index(visit, step)
            valid[visit, step] = 0.0
        for visit, step in config.interpolated_exposures:
            self._check_index(visit, step)
            interpolation[visit, step] = 1.0
        selectable = [
            (visit, step)
            for visit in range(config.visits)
            for step in range(config.local_steps)
            if valid[visit, step]
        ]
        if config.invalid_fraction:
            count = int(round(len(selectable) * config.invalid_fraction))
            for visit, step in rng.choice(selectable, size=count, replace=False) if count else ():
                valid[visit, step] = 0.0
        if config.interpolation_fraction:
            count = int(round(len(selectable) * config.interpolation_fraction))
            for visit, step in rng.choice(selectable, size=count, replace=False) if count else ():
                interpolation[visit, step] = 1.0

        mask = np.ones((config.visits, config.local_steps, len(config.wavelength_nm)), dtype=bool)
        for visit, wavelength in config.dropped_wavelengths:
            self._check_index(visit, 0)
            if not 0 <= wavelength < len(config.wavelength_nm):
                raise ValueError("dropped wavelength index out of range")
            mask[visit, :, wavelength] = False
        if config.wavelength_dropout_fraction:
            count = int(round(config.visits * len(config.wavelength_nm) * config.wavelength_dropout_fraction))
            candidates = [(v, w) for v in range(config.visits) for w in range(len(config.wavelength_nm))]
            for visit, wavelength in rng.choice(candidates, size=count, replace=False) if count else ():
                mask[visit, :, wavelength] = False
        mask &= valid[..., None].astype(bool)
        return valid, interpolation, mask

    def _latent_signals(
        self,
        mids: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
        wavelengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        config = self.config
        quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(config.quadrature_order)
        sample_times = (starts[..., None] + ends[..., None]) / 2.0 + (
            (ends - starts)[..., None] / 2.0
        ) * quadrature_nodes
        weights = quadrature_weights / 2.0
        shape = mids.shape + (len(wavelengths),)
        null = np.ones(shape, dtype=np.float64)
        injected = np.ones(shape, dtype=np.float64)
        event_mask = np.zeros(mids.shape, dtype=bool)
        depth = np.zeros(len(wavelengths), dtype=np.float64)

        if config.event_type == "transit":
            normalized_wavelength = (wavelengths - wavelengths.mean()) / max(
                np.ptp(wavelengths), 1.0
            )
            radius = config.transit_radius_ratio * (1.0 + config.wavelength_radius_slope * normalized_wavelength)
            duration_days = config.transit_duration_hours / 24.0
            epoch = config.start_bjd_tdb + config.transit_epoch_offset_days
            for wavelength_index, radius_ratio in enumerate(radius):
                delta = self._periodic_delta(sample_times, epoch, config.transit_period_days)
                velocity = 2.0 * np.sqrt(
                    max((1.0 + radius_ratio) ** 2 - config.transit_impact_parameter**2, 1e-12)
                ) / duration_days
                separation = np.sqrt(
                    config.transit_impact_parameter**2 + (velocity * delta) ** 2
                )
                drop = self._circle_overlap_fraction(separation, radius_ratio)
                averaged_drop = np.sum(drop * weights, axis=-1)
                injected[..., wavelength_index] = 1.0 - averaged_drop
                depth[wavelength_index] = float(np.max(averaged_drop))
                event_mask |= np.max(drop, axis=-1) > 1e-8
        else:
            epoch = config.start_bjd_tdb + config.microlensing_epoch_offset_days
            delta = self._periodic_delta(sample_times, epoch, config.transit_period_days)
            u = np.sqrt(config.microlensing_u0**2 + (delta / config.microlensing_timescale_days) ** 2)
            magnification = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
            averaged_magnification = np.sum(magnification * weights, axis=-1)
            injected[...] = averaged_magnification[..., None]
            depth[:] = np.max(averaged_magnification - 1.0)
            event_mask = np.max(averaged_magnification, axis=-1) > 1.0 + 1e-6
        return null, injected, event_mask, depth

    def _apply_noise(
        self,
        rng: np.random.Generator,
        latent_null: np.ndarray,
        latent_injected: np.ndarray,
        exposure_days: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        config = self.config
        count_scale = config.source_rate_per_second * config.exposure_seconds
        background = config.background_rate_per_second * config.exposure_seconds
        variability = np.empty(latent_null.shape[:2], dtype=np.float64)
        previous = 0.0
        for index in np.ndindex(variability.shape):
            previous = config.variability_ar1 * previous + rng.normal(0.0, config.variability_sigma)
            variability[index] = previous
        chromatic = 1.0 + 0.05 * self._normalize_log_wavelength(np.asarray(config.wavelength_nm))
        baseline = 1.0 + variability[..., None] * chromatic
        expected_counts = np.maximum(count_scale * baseline + background, 1.0)
        poisson_residual = rng.poisson(expected_counts) - expected_counts
        read_residual = rng.normal(0.0, config.read_noise_electrons, size=latent_null.shape)
        shared_noise = (poisson_residual + read_residual) / count_scale
        null = baseline + shared_noise
        injected = latent_injected * baseline + shared_noise
        uncertainty = np.sqrt(expected_counts + config.read_noise_electrons**2) / count_scale
        return null, injected, uncertainty

    def _render_raster(
        self,
        measurements: np.ndarray,
        uncertainty: np.ndarray,
        valid: np.ndarray,
        interpolation: np.ndarray,
        wavelength_mask: np.ndarray,
        pixel_noise: np.ndarray,
    ) -> np.ndarray:
        config = self.config
        height, width = config.raster_height, config.raster_width
        y, x = np.mgrid[0:height, 0:width]
        center_x = config.source_x * (width - 1)
        center_y = config.source_y * (height - 1)
        psf = np.exp(-0.5 * (((x - center_x) / 1.3) ** 2 + ((y - center_y) / 1.3) ** 2))
        psf = psf / psf.max()
        baseline = 1.0 + config.source_contrast * psf
        scalar = measurements.mean(axis=-1)
        scalar_uncertainty = uncertainty.mean(axis=-1)
        physical = 1.0 + config.source_contrast * psf[None, None] * scalar[..., None, None] + pixel_noise
        residual = (physical - baseline[None, None]) / np.maximum(
            scalar_uncertainty[..., None, None] + 0.0002, 1e-8
        )
        uncertainty_map = np.broadcast_to(
            scalar_uncertainty[..., None, None] + 0.0002, physical.shape
        ).copy()
        valid_map = np.broadcast_to(valid[..., None, None], physical.shape).copy()
        interpolation_map = np.broadcast_to(interpolation[..., None, None], physical.shape).copy()
        coverage = valid_map * (0.5 + 0.5 * wavelength_mask.all(axis=-1)[..., None, None])
        raster = np.stack(
            (physical, residual, uncertainty_map, valid_map, interpolation_map, coverage), axis=2
        ).astype(np.float32)
        raster *= np.where(valid_map[:, :, None], 1.0, np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)[None, None, :, None, None])
        return raster

    def _wavelength_tokens(
        self,
        measurements: np.ndarray,
        uncertainty: np.ndarray,
        coordinate: np.ndarray,
        valid: np.ndarray,
        interpolation: np.ndarray,
        wavelength_mask: np.ndarray,
    ) -> np.ndarray:
        config = self.config
        normalized_bandwidth = config.wavelength_bandwidth_nm / max(np.ptp(config.wavelength_nm), 1.0)
        normalized_log_exposure = np.log10(config.exposure_seconds) / 5.0
        residual = (measurements - 1.0) / np.maximum(uncertainty, 1e-8)
        tokens = np.stack(
            (
                np.broadcast_to(coordinate, measurements.shape),
                measurements,
                residual,
                uncertainty,
                np.full_like(measurements, normalized_bandwidth),
                np.full_like(measurements, normalized_log_exposure),
                np.ones_like(measurements),
                np.broadcast_to(valid[..., None], measurements.shape),
            ),
            axis=-1,
        ).astype(np.float32)
        tokens[..., 7] = np.broadcast_to(valid[..., None], measurements.shape)
        tokens[~wavelength_mask] = 0.0
        return tokens

    def _objects(self) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
        config = self.config
        objects = np.array(
            [
                [0.0, 0.0, 0.0, 0.02, 1.0, 0.0, 0.01, 0.2, 0.0, 0.0, 1.0, 0.0],
                [0.18, -0.12, 0.22, 0.04, 0.45, 0.2, 0.02, 0.4, 0.01, 0.1, 0.6, 0.0],
                [-0.31, 0.21, 0.37, 0.08, 0.22, -0.1, 0.04, 0.7, 0.02, -0.2, 0.3, 0.0],
            ],
            dtype=np.float32,
        )
        objects[0, 0] = config.source_x
        objects[0, 1] = config.source_y
        mask = np.ones((config.visits, objects.shape[0]), dtype=bool)
        tiled = np.broadcast_to(objects[None, ...], (config.visits, *objects.shape)).copy()
        metadata = (
            {"object_id": "synthetic-host-0001", "role": "target_host", "mass_solar": 1.0},
            {"object_id": "synthetic-neighbor-0001", "role": "foreground_context", "mass_solar": 0.6},
            {"object_id": "synthetic-neighbor-0002", "role": "background_context", "mass_solar": 0.3},
        )
        return tiled, mask, metadata

    def _context(
        self,
        starts: np.ndarray,
        mids: np.ndarray,
        ends: np.ndarray,
        valid: np.ndarray,
        interpolation: np.ndarray,
        wavelength_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        config = self.config
        visits, steps = mids.shape
        delta = np.diff(mids, axis=1, prepend=mids[:, :1])
        event_epoch = config.start_bjd_tdb + (
            config.transit_epoch_offset_days
            if config.event_type == "transit"
            else config.microlensing_epoch_offset_days
        )
        geometry_base = np.array([1.0, 0.0, 0.0, 0.0, 0.017, 0.0, 180.0 / 180.0, 20.0 / 90.0, 0.0, 0.0])
        geometry = np.broadcast_to(geometry_base, (visits, steps, 10)).copy().astype(np.float32)
        geometry[..., 9] = (mids - event_epoch).astype(np.float32)
        exposure_fraction = np.clip((ends - starts) / config.local_step_spacing_days, 0.0, 1.0)
        wavelength_fraction = wavelength_mask.mean(axis=-1)
        coverage = np.stack(
            (
                valid,
                interpolation,
                valid * (1.0 - 0.5 * interpolation),
                np.broadcast_to(wavelength_fraction, mids.shape),
                np.broadcast_to((mids.max() - mids.min()) / max(config.visit_spacing_days, 1e-8), mids.shape),
                exposure_fraction,
            ),
            axis=-1,
        ).astype(np.float32)
        local_time = np.stack(
            (
                ((mids - mids[:, :1]) / max(config.local_step_spacing_days * max(steps - 1, 1), 1e-8)),
                delta,
                starts - mids[:, :1],
                np.broadcast_to(ends - starts, mids.shape),
                mids - event_epoch,
            ),
            axis=-1,
        ).astype(np.float32)
        visit_delta = np.diff(mids[:, 0], prepend=mids[0, 0])
        long_time = np.stack(
            (
                mids[:, 0] - config.start_bjd_tdb,
                visit_delta,
                np.arange(visits, dtype=np.float64),
                np.full(visits, config.visit_spacing_days),
                np.full(visits, wavelength_fraction.mean()),
            ),
            axis=-1,
        ).astype(np.float32)
        return geometry, coverage, local_time, long_time

    def _labels(
        self,
        event_mask: np.ndarray | None,
        depth: np.ndarray,
        latent_positive: bool,
        injection_seed: int | None,
        mids: np.ndarray,
    ) -> EventLabels:
        config = self.config
        epoch = config.start_bjd_tdb + (
            config.transit_epoch_offset_days
            if config.event_type == "transit"
            else config.microlensing_epoch_offset_days
        )
        return EventLabels(
            event_type=config.event_type if latent_positive else "null",
            source_id="synthetic-host-0001",
            event_mask=event_mask,
            latent_positive=latent_positive,
            injection_seed=injection_seed,
            transit_depth=depth.astype(np.float32),
            event_midpoint_bjd_tdb=float(epoch),
            event_duration_days=float(
                config.transit_duration_hours / 24.0
                if config.event_type == "transit"
                else config.microlensing_timescale_days
            ),
            microlensing_solver_tier=(
                "not_applicable" if config.event_type == "transit" else "point_lens_analytic"
            ),
            parameter_constraint_status="unconstrained" if not latent_positive else "weakly_constrained",
        )

    @staticmethod
    def _normalize_log_wavelength(wavelengths: np.ndarray) -> np.ndarray:
        logs = np.log10(wavelengths)
        span = max(float(np.ptp(logs)), 1e-12)
        return (logs - logs.min()) / span

    @staticmethod
    def _periodic_delta(times: np.ndarray, epoch: float, period: float) -> np.ndarray:
        return (times - epoch + 0.5 * period) % period - 0.5 * period

    @staticmethod
    def _circle_overlap_fraction(separation: np.ndarray, radius: float) -> np.ndarray:
        """Area of overlap between a unit star and planet disk / star area."""

        z = np.asarray(separation, dtype=np.float64)
        k = float(radius)
        result = np.zeros_like(z)
        inside_planet = z <= max(1.0 - k, 0.0)
        result[inside_planet] = k * k
        no_overlap = z >= 1.0 + k
        partial = ~(inside_planet | no_overlap)
        zp = z[partial]
        if np.any(partial):
            cosine_star = np.clip((zp**2 + 1.0 - k**2) / (2.0 * zp), -1.0, 1.0)
            cosine_planet = np.clip((zp**2 + k**2 - 1.0) / (2.0 * zp * k), -1.0, 1.0)
            radical = np.sqrt(
                np.maximum(
                    (-zp + 1.0 + k)
                    * (zp + 1.0 - k)
                    * (zp - 1.0 + k)
                    * (zp + 1.0 + k),
                    0.0,
                )
            )
            area = np.arccos(cosine_star) + k * k * np.arccos(cosine_planet) - 0.5 * radical
            result[partial] = area / np.pi
        return np.clip(result, 0.0, 1.0)

    def _check_index(self, visit: int, step: int) -> None:
        if not (0 <= visit < self.config.visits and 0 <= step < self.config.local_steps):
            raise ValueError("exposure index out of range")
