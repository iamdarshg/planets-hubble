from __future__ import annotations

import numpy as np

from synthetic import SyntheticConfig, SyntheticGenerator


def nuisance_config(**overrides: object) -> SyntheticConfig:
    values: dict[str, object] = {
        "seed": 41,
        "visits": 3,
        "local_steps": 8,
        "raster_height": 16,
        "raster_width": 20,
        "wavelength_nm": (350.0, 650.0, 1100.0),
        "timestamp_jitter_days": 0.0,
        "visit_spacing_days": 1.0,
        "local_step_spacing_days": 0.02,
        "exposure_seconds": 120.0,
        "variability_sigma": 0.0,
        "hst_thermal_breathing_amplitude": 0.02,
        "hst_focus_psf_amplitude": 0.1,
        "pointing_jitter_pixels": 0.15,
        "drift_pixels_per_visit": 0.2,
        "roll_amplitude_deg": 1.0,
        "aberration_amplitude": 0.01,
        "geometric_distortion_amplitude": 0.01,
        "pam_gradient_amplitude": 0.02,
        "uvis_cte_loss_fraction": 0.01,
        "radiation_hot_pixel_rate": 0.02,
        "ir_persistence_amplitude": 0.01,
        "ir_nonlinearity_amplitude": 0.01,
        "cosmic_ray_rate": 0.02,
        "shutter_artifact_amplitude": 0.01,
        "kepler_quarterly_roll_amplitude": 0.02,
        "kepler_pointing_amplitude": 0.01,
        "kepler_thermal_amplitude": 0.01,
        "kepler_impulsive_rate": 0.02,
        "cbv_common_mode_amplitude": 0.02,
        "barycentric_tdb_offset_seconds": 2.0,
        "light_time_correction_seconds": 0.5,
        "apparent_position_shift_arcsec": 0.2,
    }
    values.update(overrides)
    return SyntheticConfig(**values)


def test_domain_randomized_nuisances_are_labeled_and_shared_by_pair() -> None:
    bundle = SyntheticGenerator(nuisance_config()).generate()

    expected = {
        "hst_thermal_breathing",
        "hst_focus_psf",
        "pointing_jitter",
        "pointing_drift",
        "roll",
        "aberration",
        "geometric_distortion",
        "pixel_area_map",
        "uvis_cte_loss",
        "radiation_hot_pixels",
        "ir_persistence",
        "ir_nonlinearity",
        "cosmic_ray",
        "shutter",
        "kepler_quarterly_roll",
        "kepler_pointing",
        "kepler_thermal",
        "kepler_impulsive",
        "cbv_common_mode",
    }
    assert expected.issubset(bundle.nuisance_layers)
    assert expected.issubset(bundle.nuisance_metadata)
    assert all(
        bundle.nuisance_metadata[name]["kind"] == "instrument_or_sampling_nuisance"
        for name in expected
    )
    assert np.max(np.abs(bundle.nuisance_layers["cosmic_ray"])) > 0.0
    assert np.max(np.abs(bundle.nuisance_layers["radiation_hot_pixels"])) > 0.0
    np.testing.assert_array_equal(
        bundle.null.raster[:, :, 3:, :, :], bundle.injected.raster[:, :, 3:, :, :]
    )
    assert np.all(np.isfinite(bundle.injected.raster))


def test_relativistic_terms_are_separate_from_detector_nuisance_labels() -> None:
    bundle = SyntheticGenerator(nuisance_config()).generate()

    assert set(bundle.relativity_terms) == {
        "barycentric_tdb_offset_seconds",
        "light_time_correction_seconds",
        "apparent_position_shift_arcsec",
    }
    assert bundle.relativity_metadata["time_system"] == "BJD_TDB"
    assert bundle.relativity_metadata["implementation"] == "constant_analytic_terms"
    assert bundle.relativity_metadata["detector_losses_are_relativistic"] is False
    assert np.all(np.isfinite(bundle.relativity_terms["barycentric_tdb_offset_seconds"]))
    assert bundle.relativity_terms["apparent_position_shift_arcsec"].shape[-1] == 2


def test_zero_nuisance_configuration_preserves_zero_layers_and_schedule_contract() -> None:
    config = nuisance_config(
        hst_thermal_breathing_amplitude=0.0,
        hst_focus_psf_amplitude=0.0,
        pointing_jitter_pixels=0.0,
        drift_pixels_per_visit=0.0,
        roll_amplitude_deg=0.0,
        aberration_amplitude=0.0,
        geometric_distortion_amplitude=0.0,
        pam_gradient_amplitude=0.0,
        uvis_cte_loss_fraction=0.0,
        radiation_hot_pixel_rate=0.0,
        ir_persistence_amplitude=0.0,
        ir_nonlinearity_amplitude=0.0,
        cosmic_ray_rate=0.0,
        shutter_artifact_amplitude=0.0,
        kepler_quarterly_roll_amplitude=0.0,
        kepler_pointing_amplitude=0.0,
        kepler_thermal_amplitude=0.0,
        kepler_impulsive_rate=0.0,
        cbv_common_mode_amplitude=0.0,
        barycentric_tdb_offset_seconds=0.0,
        light_time_correction_seconds=0.0,
        apparent_position_shift_arcsec=0.0,
    )
    bundle = SyntheticGenerator(config).generate()

    assert all(np.all(layer == 0.0) for layer in bundle.nuisance_layers.values())
    np.testing.assert_array_equal(
        bundle.timestamps_mid_bjd_tdb[0],
        np.arange(config.local_steps, dtype=np.float64) * config.local_step_spacing_days
        + config.start_bjd_tdb,
    )
