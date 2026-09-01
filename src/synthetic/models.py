"""Contracts for small, deterministic synthetic observation bundles.

The arrays intentionally use NumPy and omit a batch axis. ``SyntheticBundle``
can add the batch axis required by ``AstroMambaHInputs`` without importing
PyTorch or allocating a full-size 720p sequence by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SyntheticConfig:
    """Small-bundle generation parameters.

    ``raster_height`` and ``raster_width`` are deliberately small by default.
    Set them to ``720`` and ``1280`` only for an explicit model-input render.
    The local generator never creates a long 720p sequence implicitly.
    """

    seed: int = 0
    visits: int = 4
    local_steps: int = 8
    raster_height: int = 32
    raster_width: int = 48
    wavelength_nm: Tuple[float, ...] = (450.0, 650.0, 1000.0)
    wavelength_bandwidth_nm: float = 50.0
    start_bjd_tdb: float = 2460000.0
    visit_spacing_days: float = 3.0
    local_step_spacing_days: float = 0.02
    timestamp_jitter_days: float = 0.001
    exposure_seconds: float = 600.0
    quadrature_order: int = 16
    transit_period_days: float = 2.5
    transit_epoch_offset_days: float = 0.06
    transit_duration_hours: float = 3.0
    transit_radius_ratio: float = 0.1
    transit_impact_parameter: float = 0.0
    wavelength_radius_slope: float = 0.01
    source_x: float = 0.5
    source_y: float = 0.5
    source_contrast: float = 8.0
    # A compact field of unresolved sources.  Star zero is the supervised
    # target; additional stars are realistic spatial distractors and are
    # represented in both the raster and object-token context.
    field_star_count: int = 1
    field_planet_probability: float = 0.0
    field_star_flux_ratio_min: float = 0.03
    field_star_flux_ratio_max: float = 0.30
    field_star_min_separation_pixels: float = 3.0
    source_rate_per_second: float = 50000.0
    background_rate_per_second: float = 1000.0
    read_noise_electrons: float = 4.0
    # Frame-to-frame stellar brightness variation applied to the resolved
    # scene.  This is shared by the null/injected counterfactual pair, while
    # each star receives an independent temporally correlated realization.
    # The base innovation is intentionally stronger than detector shot noise:
    # it represents unresolved stellar activity/flicker.  Per-star scatter
    # below creates a quiet-to-active population rather than identical stars.
    stellar_brightness_noise_sigma: float = 0.015
    stellar_brightness_ar1: float = 0.72
    stellar_brightness_amplitude_scatter: float = 0.55
    variability_sigma: float = 0.0015
    variability_ar1: float = 0.85
    invalid_fraction: float = 0.0
    interpolation_fraction: float = 0.0
    wavelength_dropout_fraction: float = 0.0
    invalid_exposures: Tuple[Tuple[int, int], ...] = ()
    interpolated_exposures: Tuple[Tuple[int, int], ...] = ()
    dropped_wavelengths: Tuple[Tuple[int, int], ...] = ()
    event_type: str = "transit"
    microlensing_u0: float = 0.1
    microlensing_timescale_days: float = 10.0
    microlensing_epoch_offset_days: float = 0.06
    spot_rotation_period_days: float = 12.0
    spot_epoch_offset_days: float = 0.06
    spot_amplitude: float = 0.01
    spot_wavelength_slope: float = 0.2
    flare_epoch_offset_days: float = 0.12
    flare_amplitude: float = 0.2
    flare_rise_days: float = 0.01
    flare_decay_days: float = 0.05
    flare_wavelength_slope: float = 0.5
    binary_period_days: float = 5.0
    binary_epoch_offset_days: float = 0.06
    binary_duration_hours: float = 3.0
    binary_impact_parameter: float = 0.0
    binary_secondary_radius_ratio: float = 0.3
    binary_secondary_flux_ratio: float = 0.2
    binary_secondary_spectral_slope: float = 0.0
    hst_thermal_breathing_amplitude: float = 0.0
    hst_focus_psf_amplitude: float = 0.0
    pointing_jitter_pixels: float = 0.0
    drift_pixels_per_visit: float = 0.0
    roll_amplitude_deg: float = 0.0
    aberration_amplitude: float = 0.0
    geometric_distortion_amplitude: float = 0.0
    pam_gradient_amplitude: float = 0.0
    uvis_cte_loss_fraction: float = 0.0
    radiation_hot_pixel_rate: float = 0.0
    ir_persistence_amplitude: float = 0.0
    ir_nonlinearity_amplitude: float = 0.0
    cosmic_ray_rate: float = 0.0
    shutter_artifact_amplitude: float = 0.0
    kepler_quarterly_roll_amplitude: float = 0.0
    kepler_pointing_amplitude: float = 0.0
    kepler_thermal_amplitude: float = 0.0
    kepler_impulsive_rate: float = 0.0
    cbv_common_mode_amplitude: float = 0.0
    barycentric_tdb_offset_seconds: float = 0.0
    light_time_correction_seconds: float = 0.0
    apparent_position_shift_arcsec: float = 0.0
    stellar_radial_velocity_mps: float = 0.0
    barycentric_radial_velocity_mps: float = 0.0
    gravitational_redshift_mps: float = 0.0
    orbital_radial_velocity_amplitude_mps: float = 0.0
    orbital_radial_velocity_period_days: float = 2.5
    orbital_radial_velocity_phase_rad: float = 0.0

    def __post_init__(self) -> None:
        positive_ints = (
            self.visits,
            self.local_steps,
            self.raster_height,
            self.raster_width,
            self.field_star_count,
        )
        if any(value < 1 for value in positive_ints):
            raise ValueError("visits, local_steps, and raster dimensions must be positive")
        if not self.wavelength_nm or any(value <= 0 for value in self.wavelength_nm):
            raise ValueError("wavelength_nm must contain positive values")
        if any(
            value <= 0
            for value in (
                self.wavelength_bandwidth_nm,
                self.local_step_spacing_days,
                self.exposure_seconds,
                self.transit_period_days,
                self.transit_duration_hours,
                self.transit_radius_ratio,
                self.source_rate_per_second,
                self.background_rate_per_second,
                self.spot_rotation_period_days,
                self.flare_rise_days,
                self.flare_decay_days,
                self.binary_period_days,
                self.binary_duration_hours,
            self.binary_secondary_radius_ratio,
            self.orbital_radial_velocity_period_days,
        )
        ):
            raise ValueError("physical scales must be positive")
        if self.spot_amplitude < 0.0 or self.flare_amplitude < 0.0:
            raise ValueError("spot and flare amplitudes must be non-negative")
        if self.stellar_brightness_noise_sigma < 0.0:
            raise ValueError("stellar_brightness_noise_sigma must be non-negative")
        if not 0.0 <= self.stellar_brightness_ar1 < 1.0:
            raise ValueError("stellar_brightness_ar1 must be in [0, 1)")
        if self.stellar_brightness_amplitude_scatter < 0.0:
            raise ValueError("stellar_brightness_amplitude_scatter must be non-negative")
        if self.binary_secondary_flux_ratio < 0.0:
            raise ValueError("binary_secondary_flux_ratio must be non-negative")
        if not 0.0 <= self.field_planet_probability <= 1.0:
            raise ValueError("field_planet_probability must be in [0, 1]")
        if self.field_star_flux_ratio_min <= 0.0 or self.field_star_flux_ratio_max < self.field_star_flux_ratio_min:
            raise ValueError("field star flux-ratio bounds must be positive and ordered")
        if self.field_star_min_separation_pixels < 0.0:
            raise ValueError("field_star_min_separation_pixels must be non-negative")
        nonnegative_nuisances = (
            self.hst_thermal_breathing_amplitude,
            self.hst_focus_psf_amplitude,
            self.pointing_jitter_pixels,
            self.drift_pixels_per_visit,
            self.roll_amplitude_deg,
            self.aberration_amplitude,
            self.geometric_distortion_amplitude,
            self.pam_gradient_amplitude,
            self.uvis_cte_loss_fraction,
            self.radiation_hot_pixel_rate,
            self.ir_persistence_amplitude,
            self.ir_nonlinearity_amplitude,
            self.cosmic_ray_rate,
            self.shutter_artifact_amplitude,
            self.kepler_quarterly_roll_amplitude,
            self.kepler_pointing_amplitude,
            self.kepler_thermal_amplitude,
            self.kepler_impulsive_rate,
            self.cbv_common_mode_amplitude,
            self.orbital_radial_velocity_amplitude_mps,
        )
        if any(value < 0.0 for value in nonnegative_nuisances):
            raise ValueError("nuisance amplitudes and rates must be non-negative")
        for value in (
            self.stellar_radial_velocity_mps,
            self.barycentric_radial_velocity_mps,
            self.gravitational_redshift_mps,
        ):
            if not np.isfinite(value):
                raise ValueError("radial-velocity and gravitational-redshift terms must be finite")
        for rate in (
            self.radiation_hot_pixel_rate,
            self.cosmic_ray_rate,
            self.kepler_impulsive_rate,
        ):
            if rate > 1.0:
                raise ValueError("sparse nuisance rates must be in [0, 1]")
        if self.quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        if not 0.0 <= self.transit_impact_parameter < 1.0:
            raise ValueError("transit_impact_parameter must be in [0, 1)")
        if not 0.0 <= self.binary_impact_parameter < 1.0:
            raise ValueError("binary_impact_parameter must be in [0, 1)")
        for fraction in (
            self.invalid_fraction,
            self.interpolation_fraction,
            self.wavelength_dropout_fraction,
        ):
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("dropout fractions must be in [0, 1]")
        if self.event_type not in {
            "transit",
            "stellar_microlensing",
            "stellar_spot_modulation",
            "flare",
            "eclipsing_binary",
        }:
            raise ValueError(
                "event_type must be transit, stellar_microlensing, "
                "stellar_spot_modulation, flare, or eclipsing_binary"
            )


@dataclass
class EventLabels:
    """Truth and observability labels for one observation view."""

    event_type: str
    source_id: str
    event_mask: Optional[np.ndarray]
    latent_positive: bool
    injection_seed: Optional[int]
    transit_depth: np.ndarray
    event_midpoint_bjd_tdb: float
    event_duration_days: float
    microlensing_solver_tier: str
    parameter_constraint_status: str = "weakly_constrained"
    event_semantics: str = "midpoint"


@dataclass
class ObservationView:
    """Normalized raster and wavelength measurements for one paired view."""

    raster: np.ndarray
    wavelength_tokens: np.ndarray
    wavelength_mask: np.ndarray
    labels: EventLabels

    @property
    def nbytes(self) -> int:
        return int(
            self.raster.nbytes + self.wavelength_tokens.nbytes + self.wavelength_mask.nbytes
        )


@dataclass
class SyntheticBundle:
    """A null/injected pair plus common temporal, geometry, and object context."""

    null: ObservationView
    injected: ObservationView
    timestamps_start_bjd_tdb: np.ndarray
    timestamps_mid_bjd_tdb: np.ndarray
    timestamps_end_bjd_tdb: np.ndarray
    exposure_duration_seconds: np.ndarray
    wavelength_nm: np.ndarray
    object_tokens: np.ndarray
    object_mask: np.ndarray
    geometry: np.ndarray
    coverage_vector: np.ndarray
    local_time: np.ndarray
    long_time: np.ndarray
    source_metadata: dict[str, object] = field(default_factory=dict)
    object_metadata: Tuple[dict[str, object], ...] = ()
    labels: EventLabels | None = None
    nuisance_layers: dict[str, np.ndarray] = field(default_factory=dict)
    nuisance_metadata: dict[str, dict[str, object]] = field(default_factory=dict)
    relativity_terms: dict[str, np.ndarray] = field(default_factory=dict)
    relativity_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        arrays = (
            self.timestamps_start_bjd_tdb,
            self.timestamps_mid_bjd_tdb,
            self.timestamps_end_bjd_tdb,
            self.exposure_duration_seconds,
            self.wavelength_nm,
            self.object_tokens,
            self.object_mask,
            self.geometry,
            self.coverage_vector,
            self.local_time,
            self.long_time,
        )
        return int(
            self.null.nbytes
            + self.injected.nbytes
            + sum(array.nbytes for array in arrays)
            + sum(array.nbytes for array in self.nuisance_layers.values())
            + sum(array.nbytes for array in self.relativity_terms.values())
        )

    def as_model_numpy(self, view: str = "injected") -> dict[str, np.ndarray]:
        """Return a batch-one mapping matching ``AstroMambaHInputs`` shapes."""

        if view not in {"null", "injected"}:
            raise ValueError("view must be null or injected")
        selected = self.null if view == "null" else self.injected
        return {
            "raster": selected.raster[None, ...].astype(np.float32, copy=False),
            "wavelength_tokens": selected.wavelength_tokens[None, ...].astype(
                np.float32, copy=False
            ),
            "wavelength_mask": selected.wavelength_mask[None, ...],
            "object_tokens": self.object_tokens[None, ...].astype(np.float32, copy=False),
            "object_mask": self.object_mask[None, ...],
            "geometry": self.geometry[None, ...].astype(np.float32, copy=False),
            "exposure_duration": self.exposure_duration_seconds[None, ..., None].astype(
                np.float32, copy=False
            ),
            "coverage_vector": self.coverage_vector[None, ...].astype(np.float32, copy=False),
            "local_time": self.local_time[None, ...].astype(np.float32, copy=False),
            "long_time": self.long_time[None, ...].astype(np.float32, copy=False),
        }
