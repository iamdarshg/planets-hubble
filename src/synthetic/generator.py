"""Deterministic, bounded synthetic observation generation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .models import EventLabels, ObservationView, SyntheticBundle, SyntheticConfig


class SyntheticGenerator:
    """Generate paired null/injected observation bundles with NumPy only.

    The generator models an unresolved source. Its transit path is a uniform
    stellar disk and circular occultor using the analytic two-circle overlap
    area. Spot, flare, and eclipsing-binary branches are bounded analytic
    approximations and are labeled as such; they are not full stellar-surface,
    binary-star, or planetary-microlensing solvers.
    """

    RASTER_CHANNELS = (
        "physical_ratio_flux",
        "noise_scaled_residual",
        "normalized_uncertainty",
        "validity_mask",
        "interpolation_mask",
        "exposure_coverage_fraction",
    )
    SPEED_OF_LIGHT_MPS = 299_792_458.0
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
        rest_wavelength_coordinate = self._normalize_log_wavelength(wavelengths)
        valid_exposure, interpolation, wavelength_mask = self._masks(rng)

        field_stars = self._field_stars(rng)
        brightness_factors, brightness_scales = self._stellar_brightness_factors(rng)
        field_stars["brightness_factors"] = brightness_factors
        field_stars["brightness_noise_scales"] = brightness_scales
        latent_null, latent_injected, event_mask, depth = self._latent_signals(
            mids, starts, ends, wavelengths
        )
        relativity_terms, relativity_metadata = self._relativity_terms(mids)
        log_wavelength_span = max(float(np.ptp(np.log10(wavelengths))), 1e-12)
        observed_wavelength_coordinate = rest_wavelength_coordinate[None, None, :] + (
            np.log10(relativity_terms["doppler_factor"])[..., None] / log_wavelength_span
        )
        nuisance_layers = self._nuisance_layers(rng, mids, starts, ends)
        noisy_null, noisy_injected, uncertainty = self._apply_noise(
            rng, latent_null, latent_injected, exposure_days, nuisance_layers
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
            nuisance_layers,
            field_stars,
        )
        injected_raster = self._render_raster(
            noisy_injected,
            uncertainty,
            valid_exposure,
            interpolation,
            wavelength_mask,
            pixel_noise,
            nuisance_layers,
            field_stars,
        )
        null_tokens = self._wavelength_tokens(
            noisy_null,
            uncertainty,
            observed_wavelength_coordinate,
            valid_exposure,
            interpolation,
            wavelength_mask,
        )
        injected_tokens = self._wavelength_tokens(
            noisy_injected,
            uncertainty,
            observed_wavelength_coordinate,
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
        objects, object_mask, object_metadata = self._objects(field_stars)
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
            "provenance": "analytic_synthetic_generator",
            "parent_observation": None,
            "realism_tier": "R0-R3",
            "parent_conditioned": False,
            "nuisance_labels_explicit": True,
            "time_system": "BJD_TDB",
            "field_star_count": len(field_stars["positions"]),
            "target_star_index": 0,
            "stellar_brightness_model": "independent per-star AR(1) frame noise shared by null/injected pair",
            "stellar_brightness_noise_sigma": config.stellar_brightness_noise_sigma,
            "stellar_brightness_ar1": config.stellar_brightness_ar1,
            "stellar_brightness_amplitude_scatter": config.stellar_brightness_amplitude_scatter,
            "stellar_brightness_factor_std": float(np.std(field_stars["brightness_factors"])),
            "stellar_brightness_factor_min": float(np.min(field_stars["brightness_factors"])),
            "stellar_brightness_factor_max": float(np.max(field_stars["brightness_factors"])),
            "stellar_brightness_noise_scale_min": float(np.min(field_stars["brightness_noise_scales"])),
            "stellar_brightness_noise_scale_max": float(np.max(field_stars["brightness_noise_scales"])),
            "field_stars": [
                {
                    "star_index": int(index),
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "flux_ratio": float(field_stars["flux_ratios"][index]),
                    "has_exoplanet": bool(field_stars["planet_hosts"][index]),
                }
                for index, position in enumerate(field_stars["positions"])
            ],
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
            nuisance_layers=nuisance_layers,
            nuisance_metadata=self._nuisance_metadata(),
            relativity_terms=relativity_terms,
            relativity_metadata=relativity_metadata,
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
        correction_days = (
            self.config.barycentric_tdb_offset_seconds
            + self.config.light_time_correction_seconds
        ) / 86400.0
        mids += correction_days
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
        elif config.event_type == "stellar_microlensing":
            epoch = config.start_bjd_tdb + config.microlensing_epoch_offset_days
            delta = self._periodic_delta(sample_times, epoch, config.transit_period_days)
            u = np.sqrt(config.microlensing_u0**2 + (delta / config.microlensing_timescale_days) ** 2)
            magnification = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
            averaged_magnification = np.sum(magnification * weights, axis=-1)
            injected[...] = averaged_magnification[..., None]
            depth[:] = np.max(averaged_magnification - 1.0)
            event_mask = np.max(averaged_magnification, axis=-1) > 1.0 + 1e-6
        elif config.event_type == "stellar_spot_modulation":
            epoch = config.start_bjd_tdb + config.spot_epoch_offset_days
            normalized_wavelength = self._centered_wavelength(wavelengths)
            chromatic_amplitude = config.spot_amplitude * (
                1.0 + config.spot_wavelength_slope * normalized_wavelength
            )
            phase = 2.0 * np.pi * (sample_times - epoch) / config.spot_rotation_period_days
            modulation = 1.0 - chromatic_amplitude[None, None, None, :] * (
                0.5 + 0.5 * np.cos(phase[..., None])
            )
            injected = np.sum(modulation * weights[..., None], axis=-2)
            drop = 1.0 - modulation
            averaged_drop = np.sum(drop * weights[..., None], axis=-2)
            depth = np.max(averaged_drop, axis=(0, 1))
            event_mask = np.max(drop, axis=(-1, -2)) > max(config.spot_amplitude * 0.01, 1e-8)
        elif config.event_type == "flare":
            epoch = config.start_bjd_tdb + config.flare_epoch_offset_days
            normalized_wavelength = self._centered_wavelength(wavelengths)
            chromatic_amplitude = config.flare_amplitude * (
                1.0 + config.flare_wavelength_slope * normalized_wavelength
            )
            delta = sample_times - epoch
            rise = np.exp(np.minimum(delta, 0.0) / config.flare_rise_days)
            decay = np.exp(-np.maximum(delta, 0.0) / config.flare_decay_days)
            flare = (
                chromatic_amplitude[None, None, None, :]
                * np.where(delta[..., None] < 0.0, rise[..., None], decay[..., None])
            )
            injected = 1.0 + np.sum(flare * weights[..., None], axis=-2)
            averaged_flare = np.sum(flare * weights[..., None], axis=-2)
            depth = np.max(averaged_flare, axis=(0, 1))
            event_mask = np.max(flare, axis=(-1, -2)) > max(config.flare_amplitude * 0.01, 1e-8)
        else:
            epoch = config.start_bjd_tdb + config.binary_epoch_offset_days
            normalized_wavelength = self._centered_wavelength(wavelengths)
            secondary_flux = config.binary_secondary_flux_ratio * (
                1.0 + config.binary_secondary_spectral_slope * normalized_wavelength
            )
            secondary_flux = np.maximum(secondary_flux, 0.0)
            total_flux = 1.0 + secondary_flux
            duration_days = config.binary_duration_hours / 24.0
            velocity = 2.0 * np.sqrt(
                max(
                    (1.0 + config.binary_secondary_radius_ratio) ** 2
                    - config.binary_impact_parameter**2,
                    1e-12,
                )
            ) / duration_days
            primary_delta = self._periodic_delta(sample_times, epoch, config.binary_period_days)
            secondary_delta = self._periodic_delta(
                sample_times, epoch + 0.5 * config.binary_period_days, config.binary_period_days
            )
            primary_separation = np.sqrt(
                config.binary_impact_parameter**2 + (velocity * primary_delta) ** 2
            )
            secondary_separation = np.sqrt(
                config.binary_impact_parameter**2 + (velocity * secondary_delta) ** 2
            )
            primary_overlap = self._circle_overlap_fraction(
                primary_separation, config.binary_secondary_radius_ratio
            )
            secondary_overlap = self._circle_overlap_fraction(
                secondary_separation, config.binary_secondary_radius_ratio
            )
            primary_loss = primary_overlap[..., None] / total_flux
            secondary_loss = secondary_overlap[..., None] * secondary_flux / total_flux
            loss = primary_loss + secondary_loss
            averaged_loss = np.sum(loss * weights[..., None], axis=-2)
            injected = 1.0 - averaged_loss
            depth = np.max(averaged_loss, axis=(0, 1))
            event_mask = np.max(loss, axis=(-1, -2)) > 1e-8
        return null, injected, event_mask, depth

    def _apply_noise(
        self,
        rng: np.random.Generator,
        latent_null: np.ndarray,
        latent_injected: np.ndarray,
        exposure_days: float,
        nuisance_layers: dict[str, np.ndarray] | None = None,
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
        flux_nuisance = self._flux_nuisance(nuisance_layers, variability.shape)
        baseline = 1.0 + (variability[..., None] * chromatic) + flux_nuisance[..., None]
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
        nuisance_layers: dict[str, np.ndarray] | None = None,
        field_stars: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        config = self.config
        height, width = config.raster_height, config.raster_width
        y, x = np.mgrid[0:height, 0:width]
        stars = field_stars or self._field_stars(np.random.default_rng(config.seed))
        layers = nuisance_layers or {}
        focus = layers.get("hst_focus_psf", np.zeros(measurements.shape[:2]))
        jitter = layers.get(
            "pointing_jitter", np.zeros((*measurements.shape[:2], 2), dtype=np.float64)
        )
        drift = layers.get(
            "pointing_drift", np.zeros((*measurements.shape[:2], 2), dtype=np.float64)
        )
        roll = layers.get("roll", np.zeros(measurements.shape[:2], dtype=np.float64))
        roll += np.rad2deg(
            layers.get("kepler_quarterly_roll", np.zeros(measurements.shape[:2], dtype=np.float64))
        )
        sigma = np.maximum(1.3 * (1.0 + focus), 0.25)
        field_center_x = 0.5 * (width - 1)
        field_center_y = 0.5 * (height - 1)
        theta = np.deg2rad(roll)
        positions = np.asarray(stars["positions"], dtype=np.float64)
        source_offset_x = positions[:, 0] * (width - 1) - field_center_x
        source_offset_y = positions[:, 1] * (height - 1) - field_center_y
        rolled_source_x = (
            field_center_x
            + np.cos(theta)[..., None] * source_offset_x[None, None, :]
            - np.sin(theta)[..., None] * source_offset_y[None, None, :]
        )
        rolled_source_y = (
            field_center_y
            + np.sin(theta)[..., None] * source_offset_x[None, None, :]
            + np.cos(theta)[..., None] * source_offset_y[None, None, :]
        )
        shifted_x = (
            x[None, None, None]
            - rolled_source_x[..., None, None]
            - jitter[..., None, 0, None, None]
            - drift[..., None, 0, None, None]
        )
        shifted_y = (
            y[None, None, None]
            - rolled_source_y[..., None, None]
            - jitter[..., None, 1, None, None]
            - drift[..., None, 1, None, None]
        )
        star_psf = np.exp(
            -0.5
            * (
                (shifted_x / sigma[..., None, None, None]) ** 2
                + (shifted_y / sigma[..., None, None, None]) ** 2
            )
        )
        star_psf /= np.maximum(star_psf.max(axis=(-1, -2), keepdims=True), 1e-8)
        brightness_factors = np.asarray(
            stars.get(
                "brightness_factors",
                np.ones((*measurements.shape[:2], len(stars["positions"])), dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if brightness_factors.shape != (*measurements.shape[:2], len(stars["positions"])):
            raise ValueError(
                "brightness_factors must be [visits, local_steps, field_star_count], "
                f"got {brightness_factors.shape}"
            )
        weighted_flux_ratios = brightness_factors * np.asarray(stars["flux_ratios"], dtype=np.float32)[None, None, :]
        field_psf = np.einsum("vls,vlshw->vlhw", weighted_flux_ratios, star_psf)
        target_psf = star_psf[:, :, 0]
        target_brightness = brightness_factors[..., 0]
        baseline = 1.0 + config.source_contrast * field_psf
        scalar = measurements.mean(axis=-1)
        scalar_uncertainty = uncertainty.mean(axis=-1)
        spatial = layers.get("pixel_area_map", np.zeros((height, width), dtype=np.float64))[None, None]
        spatial = spatial + layers.get(
            "radiation_hot_pixels", np.zeros_like(pixel_noise)
        ) + layers.get("cosmic_ray", np.zeros_like(pixel_noise))
        geometric = layers.get("geometric_distortion", np.zeros(measurements.shape[:2]))
        field_measurement = (
            target_psf * (scalar * target_brightness)[..., None, None]
            + (field_psf - target_psf * target_brightness[..., None, None])
        )
        physical = (
            1.0
            + config.source_contrast * field_measurement
            + pixel_noise
            + spatial
            + geometric[..., None, None]
        )
        residual = (physical - baseline * (1.0 + geometric[..., None, None])) / np.maximum(
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

    def _nuisance_layers(
        self,
        rng: np.random.Generator,
        mids: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Create shared, labeled nuisance layers with bounded NumPy arrays.

        These are domain-randomizable approximations intended for pretraining.
        They are not a replacement for calibration with real HST/Kepler data.
        Detector losses and throughput-like terms are instrument nuisances, not
        relativistic corrections; the latter are returned separately.
        """

        config = self.config
        shape = mids.shape
        phase = 2.0 * np.pi * (mids - config.start_bjd_tdb) / max(
            config.visit_spacing_days * max(config.visits, 1), 1e-8
        )
        layers: dict[str, np.ndarray] = {
            "hst_thermal_breathing": (
                config.hst_thermal_breathing_amplitude * np.sin(phase)
            ).astype(np.float32),
            "hst_focus_psf": (
                config.hst_focus_psf_amplitude * np.sin(phase + 0.7)
            ).astype(np.float32),
            "pointing_jitter": rng.normal(
                0.0, config.pointing_jitter_pixels, size=(*shape, 2)
            ).astype(np.float32),
            "pointing_drift": np.stack(
                (
                    config.drift_pixels_per_visit
                    * np.arange(config.visits, dtype=np.float64)[:, None]
                    / max(config.visits - 1, 1)
                    * np.ones(shape, dtype=np.float64),
                    np.zeros(shape, dtype=np.float64),
                ),
                axis=-1,
            ).astype(np.float32),
            "roll": (
                config.roll_amplitude_deg * np.sin(phase + 0.3)
            ).astype(np.float32),
            "aberration": (
                config.aberration_amplitude * np.cos(phase + 1.1)
            ).astype(np.float32),
            "geometric_distortion": (
                config.geometric_distortion_amplitude * np.sin(phase + 0.4)
            ).astype(np.float32),
            "pixel_area_map": self._pixel_area_map(config.pam_gradient_amplitude),
            "uvis_cte_loss": np.full(
                shape, config.uvis_cte_loss_fraction, dtype=np.float32
            ),
            "radiation_hot_pixels": self._sparse_hot_pixels(rng, config.radiation_hot_pixel_rate),
            "ir_persistence": np.zeros(shape, dtype=np.float32),
            "ir_nonlinearity": np.full(
                shape, config.ir_nonlinearity_amplitude, dtype=np.float32
            ),
            "cosmic_ray": self._sparse_cosmic_rays(rng, config.cosmic_ray_rate),
            "shutter": (
                config.shutter_artifact_amplitude
                * np.clip((ends - starts) / max(config.exposure_seconds / 86400.0, 1e-12), 0.0, 1.0)
            ).astype(np.float32),
            "kepler_quarterly_roll": (
                config.kepler_quarterly_roll_amplitude
                * np.sin(2.0 * np.pi * np.arange(config.visits)[:, None] / 4.0)
                * np.ones(shape, dtype=np.float64)
            ).astype(np.float32),
            "kepler_pointing": rng.normal(
                0.0, config.kepler_pointing_amplitude, size=shape
            ).astype(np.float32),
            "kepler_thermal": (
                config.kepler_thermal_amplitude * np.cos(phase / 2.0)
            ).astype(np.float32),
            "kepler_impulsive": self._impulsive_layer(rng, config.kepler_impulsive_rate),
            "cbv_common_mode": (
                config.cbv_common_mode_amplitude * np.sin(phase / 3.0 + 0.2)
            ).astype(np.float32),
        }
        persistence = layers["ir_persistence"]
        if config.ir_persistence_amplitude:
            persistence[:, 1:] = config.ir_persistence_amplitude * np.maximum(
                0.0, layers["hst_thermal_breathing"][:, :-1]
            )
        return layers

    def _flux_nuisance(
        self, layers: dict[str, np.ndarray] | None, shape: tuple[int, int]
    ) -> np.ndarray:
        if not layers:
            return np.zeros(shape, dtype=np.float64)
        return np.asarray(
            layers["hst_thermal_breathing"]
            + layers["aberration"]
            + layers["uvis_cte_loss"] * -1.0
            + layers["ir_persistence"]
            + layers["ir_nonlinearity"]
            + layers["shutter"]
            + layers["kepler_quarterly_roll"]
            + layers["kepler_pointing"]
            + layers["kepler_thermal"]
            + layers["kepler_impulsive"]
            + layers["cbv_common_mode"],
            dtype=np.float64,
        )

    def _relativity_terms(
        self, mids: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        config = self.config
        orbital_velocity = config.orbital_radial_velocity_amplitude_mps * np.sin(
            2.0 * np.pi * (mids - config.start_bjd_tdb) / config.orbital_radial_velocity_period_days
            + config.orbital_radial_velocity_phase_rad
        )
        radial_velocity = (
            config.stellar_radial_velocity_mps
            + config.barycentric_radial_velocity_mps
            + config.gravitational_redshift_mps
            + orbital_velocity
        ).astype(np.float64)
        beta = np.clip(radial_velocity / self.SPEED_OF_LIGHT_MPS, -0.999999, 0.999999)
        # Relativistic longitudinal Doppler factor. Positive radial velocity
        # is receding/redshifted; this is an analytic spectral-coordinate
        # transform, not a claim of full radiative-transfer modelling.
        doppler_factor = np.sqrt((1.0 + beta) / (1.0 - beta)).astype(np.float32)
        return (
            {
                "barycentric_tdb_offset_seconds": np.full(
                    mids.shape, config.barycentric_tdb_offset_seconds, dtype=np.float32
                ),
                "light_time_correction_seconds": np.full(
                    mids.shape, config.light_time_correction_seconds, dtype=np.float32
                ),
                "apparent_position_shift_arcsec": np.broadcast_to(
                    np.array([config.apparent_position_shift_arcsec, 0.0], dtype=np.float32),
                    (*mids.shape, 2),
                ).copy(),
                "radial_velocity_mps": radial_velocity.astype(np.float32),
                "orbital_radial_velocity_mps": orbital_velocity.astype(np.float32),
                "doppler_factor": doppler_factor,
                "doppler_shift_fraction": (doppler_factor - 1.0).astype(np.float32),
            },
            {
                "time_system": "BJD_TDB",
                "implementation": "analytic_barycentric_light_time_position_and_relativistic_doppler_terms",
                "detector_losses_are_relativistic": False,
                "doppler_applied_to": "wavelength_token_coordinate",
                "radial_velocity_sign": "positive_is_receding_redshift",
                "scope": "bounded_pretraining_approximation_requires_real_data_calibration",
            },
        )

    def _nuisance_metadata(self) -> dict[str, dict[str, object]]:
        names = (
            "hst_thermal_breathing", "hst_focus_psf", "pointing_jitter", "pointing_drift",
            "roll", "aberration", "geometric_distortion", "pixel_area_map", "uvis_cte_loss",
            "radiation_hot_pixels", "ir_persistence", "ir_nonlinearity", "cosmic_ray", "shutter",
            "kepler_quarterly_roll", "kepler_pointing", "kepler_thermal", "kepler_impulsive",
            "cbv_common_mode",
        )
        return {
            name: {
                "kind": "instrument_or_sampling_nuisance",
                "domain_randomizable": True,
                "implementation": "bounded_analytic_approximation",
                "scientific_status": "pretraining_prior_requires_real_data_calibration",
            }
            for name in names
        }

    def _pixel_area_map(self, amplitude: float) -> np.ndarray:
        y, x = np.mgrid[0 : self.config.raster_height, 0 : self.config.raster_width]
        x_norm = (x / max(self.config.raster_width - 1, 1)) - 0.5
        return (amplitude * x_norm).astype(np.float32)

    def _sparse_hot_pixels(self, rng: np.random.Generator, rate: float) -> np.ndarray:
        shape = (self.config.visits, self.config.local_steps, self.config.raster_height, self.config.raster_width)
        mask = rng.random(shape) < rate
        return (mask * -0.05).astype(np.float32)

    def _sparse_cosmic_rays(self, rng: np.random.Generator, rate: float) -> np.ndarray:
        shape = (self.config.visits, self.config.local_steps, self.config.raster_height, self.config.raster_width)
        mask = rng.random(shape) < rate
        amplitudes = rng.uniform(0.1, 0.5, size=shape)
        return (mask * amplitudes).astype(np.float32)

    def _impulsive_layer(self, rng: np.random.Generator, rate: float) -> np.ndarray:
        shape = (self.config.visits, self.config.local_steps)
        mask = rng.random(shape) < rate
        return (mask * rng.uniform(-0.02, 0.02, size=shape)).astype(np.float32)

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

    def _field_stars(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Create a deterministic multi-star field with an explicit target.

        Star zero is always the configured target.  The other sources are
        sampled away from it and receive independent brightness and planet
        host priors.  The null/injected views reuse this exact scene.
        """

        config = self.config
        positions = [(float(config.source_x), float(config.source_y))]
        minimum = config.field_star_min_separation_pixels / max(
            config.raster_width - 1, config.raster_height - 1, 1
        )
        attempts = 0
        while len(positions) < config.field_star_count and attempts < 500:
            attempts += 1
            candidate = tuple(float(value) for value in rng.uniform(0.06, 0.94, size=2))
            if all(np.hypot(candidate[0] - x0, candidate[1] - y0) >= minimum for x0, y0 in positions):
                positions.append(candidate)
        while len(positions) < config.field_star_count:
            # Extremely crowded compact scenes still get a valid bounded
            # field; the normal rejection path handles ordinary settings.
            positions.append(tuple(float(value) for value in rng.uniform(0.06, 0.94, size=2)))
        flux_ratios = np.asarray(
            [1.0]
            + [
                float(rng.uniform(config.field_star_flux_ratio_min, config.field_star_flux_ratio_max))
                for _ in range(config.field_star_count - 1)
            ],
            dtype=np.float32,
        )
        planet_hosts = np.asarray(
            [True]
            + [
                bool(rng.random() < config.field_planet_probability)
                for _ in range(config.field_star_count - 1)
            ],
            dtype=bool,
        )
        return {
            "positions": np.asarray(positions, dtype=np.float32),
            "flux_ratios": flux_ratios,
            "planet_hosts": planet_hosts,
        }

    def _stellar_brightness_factors(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Generate correlated brightness factors for a quiet-to-active star population."""

        config = self.config
        factors = np.ones(
            (config.visits, config.local_steps, config.field_star_count),
            dtype=np.float32,
        )
        scales = np.ones(config.field_star_count, dtype=np.float32)
        if config.stellar_brightness_noise_sigma == 0.0:
            return factors, scales
        scales = np.clip(
            np.exp(rng.normal(0.0, config.stellar_brightness_amplitude_scatter, size=config.field_star_count)),
            0.35,
            3.0,
        ).astype(np.float32)
        for star_index in range(config.field_star_count):
            state = 0.0
            innovation_sigma = config.stellar_brightness_noise_sigma * float(scales[star_index])
            for visit in range(config.visits):
                for step in range(config.local_steps):
                    state = (
                        config.stellar_brightness_ar1 * state
                        + rng.normal(0.0, innovation_sigma)
                    )
                    # Keep the nuisance bounded so a rare random walk cannot
                    # create an unphysical negative or saturated star.
                    factors[visit, step, star_index] = float(np.clip(1.0 + state, 0.5, 1.5))
        return factors, scales

    def _objects(self, field_stars: dict[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, tuple[dict[str, object], ...]]:
        config = self.config
        stars = field_stars or self._field_stars(np.random.default_rng(config.seed))
        positions = [tuple(position) for position in stars["positions"]]
        flux_ratios = [float(value) for value in stars["flux_ratios"]]
        planet_hosts = [bool(value) for value in stars["planet_hosts"]]
        # Preserve the original three-token context contract for small/default
        # bundles. These two entries are catalog-context tokens, not extra
        # raster sources and are kept away from the target center.
        context_positions = ((0.16, 0.82), (0.84, 0.18))
        while len(positions) < 3:
            positions.append(context_positions[len(positions) - 1])
            flux_ratios.append(0.10 if len(positions) == 2 else 0.06)
            planet_hosts.append(False)
        objects = np.zeros((len(positions), 12), dtype=np.float32)
        for index, (x0, y0) in enumerate(positions):
            flux_ratio = flux_ratios[index]
            objects[index] = np.array(
                [
                    x0,
                    y0,
                    flux_ratio,
                    0.02 + 0.01 * index,
                    flux_ratio,
                    float(planet_hosts[index]),
                    0.01 + 0.01 * index,
                    0.2 + 0.05 * index,
                    0.0,
                    0.0,
                    1.0,
                    float(index == 0),
                ],
                dtype=np.float32,
            )
        mask = np.ones((config.visits, objects.shape[0]), dtype=bool)
        tiled = np.broadcast_to(objects[None, ...], (config.visits, *objects.shape)).copy()
        metadata = tuple(
            {
                "object_id": f"synthetic-star-{index:04d}",
                "role": "target_host" if index == 0 else "field_star",
                "mass_solar": float(max(0.2, flux_ratio ** 0.25)),
                "flux_ratio": float(flux_ratio),
                "has_exoplanet": bool(planet_hosts[index]),
            }
            for index, flux_ratio in enumerate(flux_ratios)
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
        event_epoch = self._event_epoch()
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
        epoch = self._event_epoch()
        duration, semantics = self._event_duration_and_semantics()
        return EventLabels(
            event_type=config.event_type if latent_positive else "null",
            source_id="synthetic-host-0001",
            event_mask=event_mask,
            latent_positive=latent_positive,
            injection_seed=injection_seed,
            transit_depth=depth.astype(np.float32),
            event_midpoint_bjd_tdb=float(epoch),
            event_duration_days=float(duration),
            microlensing_solver_tier=(
                "point_lens_analytic"
                if config.event_type == "stellar_microlensing"
                else "not_applicable"
            ),
            parameter_constraint_status="unconstrained" if not latent_positive else "weakly_constrained",
            event_semantics="no_injection" if not latent_positive else semantics,
        )

    def _event_epoch(self) -> float:
        config = self.config
        offsets = {
            "transit": config.transit_epoch_offset_days,
            "stellar_microlensing": config.microlensing_epoch_offset_days,
            "stellar_spot_modulation": config.spot_epoch_offset_days,
            "flare": config.flare_epoch_offset_days,
            "eclipsing_binary": config.binary_epoch_offset_days,
        }
        return config.start_bjd_tdb + offsets[config.event_type]

    def _event_duration_and_semantics(self) -> tuple[float, str]:
        config = self.config
        if config.event_type == "transit":
            return config.transit_duration_hours / 24.0, "midpoint"
        if config.event_type == "stellar_microlensing":
            return config.microlensing_timescale_days, "point_lens_peak"
        if config.event_type == "stellar_spot_modulation":
            return config.spot_rotation_period_days, "persistent_modulation"
        if config.event_type == "flare":
            return config.flare_rise_days + config.flare_decay_days, "injected_peak"
        return config.binary_duration_hours / 24.0, "uniform_disk_eclipse_approximation"

    @staticmethod
    def _centered_wavelength(wavelengths: np.ndarray) -> np.ndarray:
        return (wavelengths - wavelengths.mean()) / max(np.ptp(wavelengths), 1.0)

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
