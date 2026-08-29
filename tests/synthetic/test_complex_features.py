from __future__ import annotations

import numpy as np

from synthetic import SyntheticConfig, SyntheticGenerator


def feature_config(**overrides: object) -> SyntheticConfig:
    values: dict[str, object] = {
        "seed": 29,
        "visits": 2,
        "local_steps": 16,
        "raster_height": 12,
        "raster_width": 16,
        "wavelength_nm": (400.0, 650.0, 1000.0),
        "timestamp_jitter_days": 0.0,
        "visit_spacing_days": 1.0,
        "local_step_spacing_days": 0.02,
        "exposure_seconds": 120.0,
        "source_rate_per_second": 50000.0,
        "background_rate_per_second": 1000.0,
        "variability_sigma": 0.0,
    }
    values.update(overrides)
    return SyntheticConfig(**values)


def test_spot_modulation_is_chromatic_and_explicitly_labeled() -> None:
    bundle = SyntheticGenerator(
        feature_config(
            event_type="stellar_spot_modulation",
            spot_rotation_period_days=0.8,
            spot_epoch_offset_days=0.2,
            spot_amplitude=0.04,
            spot_wavelength_slope=0.5,
        )
    ).generate()

    labels = bundle.injected.labels
    assert labels.event_type == "stellar_spot_modulation"
    assert labels.event_semantics == "persistent_modulation"
    assert labels.latent_positive
    assert labels.event_mask is not None
    assert labels.event_mask.any()
    assert labels.injection_seed == 29
    assert labels.microlensing_solver_tier == "not_applicable"

    injected = bundle.injected.wavelength_tokens[..., 1]
    null = bundle.null.wavelength_tokens[..., 1]
    signal = injected - null
    assert np.max(np.abs(signal)) > 0.01
    assert not np.allclose(signal[..., 0], signal[..., -1])
    np.testing.assert_array_equal(bundle.null.wavelength_mask, bundle.injected.wavelength_mask)


def test_flare_is_a_finite_exposure_integrated_transient_with_peak_label() -> None:
    config = feature_config(
        visits=1,
        local_steps=21,
        local_step_spacing_days=0.01,
        event_type="flare",
        flare_epoch_offset_days=0.1,
        flare_amplitude=0.25,
        flare_rise_days=0.01,
        flare_decay_days=0.04,
        flare_wavelength_slope=0.8,
    )
    bundle = SyntheticGenerator(config).generate()

    labels = bundle.labels
    assert labels is not None
    assert labels.event_type == "flare"
    assert labels.event_semantics == "injected_peak"
    assert labels.event_mask is not None
    assert labels.event_mask.any()
    peak = np.argmax(bundle.injected.wavelength_tokens[0, :, 1, 1])
    assert abs(bundle.timestamps_mid_bjd_tdb[0, peak] - labels.event_midpoint_bjd_tdb) < 0.02
    assert bundle.injected.wavelength_tokens[0, peak, -1, 1] > bundle.injected.wavelength_tokens[0, peak, 0, 1]
    assert np.all(np.isfinite(bundle.injected.raster))


def test_eclipsing_binary_approximation_has_two_labeled_eclipse_windows() -> None:
    config = feature_config(
        visits=1,
        local_steps=41,
        local_step_spacing_days=0.01,
        event_type="eclipsing_binary",
        binary_period_days=0.3,
        binary_epoch_offset_days=0.1,
        binary_duration_hours=1.5,
        binary_secondary_radius_ratio=0.3,
        binary_secondary_flux_ratio=0.2,
    )
    bundle = SyntheticGenerator(config).generate()
    labels = bundle.labels

    assert labels is not None
    assert labels.event_type == "eclipsing_binary"
    assert labels.event_semantics == "uniform_disk_eclipse_approximation"
    assert labels.microlensing_solver_tier == "not_applicable"
    assert labels.event_mask is not None
    event_indices = np.flatnonzero(labels.event_mask[0])
    assert event_indices.size > 0
    assert np.all(
        bundle.injected.wavelength_tokens[..., 1]
        <= bundle.null.wavelength_tokens[..., 1] + 1e-6
    )
    assert np.min(bundle.injected.wavelength_tokens[..., 1]) < 1.0
    assert bundle.injected.wavelength_tokens.shape[-1] == 8
    assert bundle.injected.wavelength_mask.shape == (1, 41, 3)


def test_complex_feature_pair_preserves_masks_and_null_labels() -> None:
    bundle = SyntheticGenerator(
        feature_config(
            event_type="flare",
            invalid_exposures=((0, 2),),
            interpolated_exposures=((1, 3),),
            dropped_wavelengths=((0, 1),),
        )
    ).generate()

    assert bundle.null.labels.event_type == "null"
    assert not bundle.null.labels.latent_positive
    assert bundle.null.labels.event_mask is None
    assert bundle.injected.labels.latent_positive
    np.testing.assert_array_equal(bundle.null.wavelength_mask, bundle.injected.wavelength_mask)
    np.testing.assert_array_equal(
        bundle.null.raster[:, :, 3:, :, :], bundle.injected.raster[:, :, 3:, :, :]
    )
    assert np.all(bundle.injected.raster[0, 2, 3] == 0.0)
    assert np.all(bundle.injected.raster[1, 3, 4] == 1.0)
    assert not bundle.injected.wavelength_mask[0, 0, 1]
    assert np.all(np.isfinite(bundle.timestamps_mid_bjd_tdb))
    assert np.all(np.diff(bundle.timestamps_mid_bjd_tdb[0]) > 0.0)
