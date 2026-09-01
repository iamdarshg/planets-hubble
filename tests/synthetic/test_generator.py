from __future__ import annotations

import tracemalloc

import numpy as np

from synthetic import SyntheticConfig, SyntheticGenerator


def small_config(**overrides: object) -> SyntheticConfig:
    values: dict[str, object] = {
        "visits": 3,
        "local_steps": 8,
        "raster_height": 24,
        "raster_width": 32,
        "wavelength_nm": (450.0, 650.0, 1000.0),
        "invalid_fraction": 0.0,
        "interpolation_fraction": 0.0,
        "wavelength_dropout_fraction": 0.0,
        "transit_radius_ratio": 0.1,
        "transit_duration_hours": 3.0,
        "transit_epoch_offset_days": 0.06,
        "local_step_spacing_days": 0.02,
    }
    values.update(overrides)
    return SyntheticConfig(**values)


def test_same_seed_reproduces_entire_paired_bundle() -> None:
    config = small_config(seed=17)
    first = SyntheticGenerator(config).generate()
    second = SyntheticGenerator(config).generate()

    np.testing.assert_array_equal(first.injected.raster, second.injected.raster)
    np.testing.assert_array_equal(first.null.wavelength_tokens, second.null.wavelength_tokens)
    np.testing.assert_array_equal(first.timestamps_mid_bjd_tdb, second.timestamps_mid_bjd_tdb)
    np.testing.assert_array_equal(first.labels.event_mask, second.labels.event_mask)
    assert first.labels.event_type == "transit"
    assert first.labels.injection_seed == 17
    assert first.source_metadata["realism_tier"] == "R0-R3"
    assert first.source_metadata["parent_conditioned"] is False


def test_stellar_brightness_noise_is_enabled_bounded_and_reproducible() -> None:
    config = small_config(
        seed=23,
        field_star_count=5,
        stellar_brightness_noise_sigma=0.02,
        stellar_brightness_amplitude_scatter=0.55,
    )
    first = SyntheticGenerator(config).generate()
    second = SyntheticGenerator(config).generate()

    assert first.source_metadata["stellar_brightness_model"].startswith("independent per-star")
    assert first.source_metadata["stellar_brightness_factor_std"] > 0.0
    assert first.source_metadata["stellar_brightness_noise_scale_min"] >= 0.35 - 1e-6
    assert first.source_metadata["stellar_brightness_noise_scale_max"] <= 3.0 + 1e-6
    assert 0.5 <= first.source_metadata["stellar_brightness_factor_min"] <= 1.0
    assert 1.0 <= first.source_metadata["stellar_brightness_factor_max"] <= 1.5
    assert first.source_metadata["stellar_brightness_factor_std"] == second.source_metadata["stellar_brightness_factor_std"]
    np.testing.assert_array_equal(first.null.raster, second.null.raster)

    quiet = SyntheticGenerator(
        small_config(
            seed=23,
            field_star_count=5,
            stellar_brightness_noise_sigma=0.0,
            stellar_brightness_amplitude_scatter=0.55,
        )
    ).generate()
    assert quiet.source_metadata["stellar_brightness_factor_std"] == 0.0


def test_exposure_integrated_transit_has_expected_depth_and_timing() -> None:
    config = small_config(
        visits=1,
        local_steps=13,
        exposure_seconds=120.0,
        local_step_spacing_days=0.01,
        transit_duration_hours=2.0,
        transit_radius_ratio=0.1,
    )
    bundle = SyntheticGenerator(config).generate()
    flux = bundle.injected.wavelength_tokens[0, :, 1, 1]
    expected_epoch = config.start_bjd_tdb + config.transit_epoch_offset_days
    nearest_index = int(np.argmin(np.abs(bundle.timestamps_mid_bjd_tdb[0] - expected_epoch)))

    assert abs(bundle.timestamps_mid_bjd_tdb[0, nearest_index] - expected_epoch) < 0.01
    assert 0.985 < flux[nearest_index] < 1.0
    assert bundle.labels.event_mask[0, nearest_index]
    assert bundle.labels.transit_depth[0] > 0.009


def test_masks_and_normalized_views_preserve_missingness() -> None:
    config = small_config(
        invalid_exposures=((1, 2),),
        interpolated_exposures=((0, 3),),
        dropped_wavelengths=((2, 1),),
    )
    bundle = SyntheticGenerator(config).generate()
    invalid = bundle.injected.raster[1, 2]

    assert np.all(invalid[3] == 0.0)
    assert np.all(invalid[5] == 0.0)
    assert np.all(bundle.injected.raster[0, 3, 4] == 1.0)
    assert not bundle.injected.wavelength_mask[2, 0, 1]
    assert bundle.injected.wavelength_mask[2, 0, 0]
    assert np.all(np.isfinite(bundle.injected.raster))
    assert np.all(np.isfinite(bundle.injected.wavelength_tokens))


def test_null_and_injected_pair_share_schedule_and_differ_only_by_signal_path() -> None:
    bundle = SyntheticGenerator(small_config()).generate()

    np.testing.assert_array_equal(bundle.null.raster[3:], bundle.injected.raster[3:])
    np.testing.assert_array_equal(bundle.null.wavelength_mask, bundle.injected.wavelength_mask)
    np.testing.assert_array_equal(bundle.null.wavelength_tokens[..., 0], bundle.injected.wavelength_tokens[..., 0])
    assert np.max(np.abs(bundle.injected.wavelength_tokens[..., 1] - bundle.null.wavelength_tokens[..., 1])) > 0.0
    assert bundle.null.labels.event_mask is None
    assert bundle.labels.latent_positive


def test_model_contract_arrays_have_batch_axes_and_configurable_small_spatial_raster() -> None:
    bundle = SyntheticGenerator(small_config()).generate()
    arrays = bundle.as_model_numpy()

    assert arrays["raster"].shape == (1, 3, 8, 6, 24, 32)
    assert arrays["wavelength_tokens"].shape == (1, 3, 8, 3, 8)
    assert arrays["wavelength_mask"].dtype == np.bool_
    assert arrays["object_tokens"].shape == (1, 3, 3, 12)
    assert arrays["geometry"].shape == (1, 3, 8, 10)
    assert arrays["exposure_duration"].shape == (1, 3, 8, 1)
    assert arrays["local_time"].shape == (1, 3, 8, 5)
    assert arrays["long_time"].shape == (1, 3, 5)


def test_generation_peak_python_allocation_stays_bounded() -> None:
    config = small_config(
        visits=6,
        local_steps=16,
        raster_height=48,
        raster_width=64,
        wavelength_nm=tuple(np.linspace(350.0, 1700.0, 12)),
    )
    tracemalloc.start()
    bundle = SyntheticGenerator(config).generate()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert bundle.nbytes < 32 * 1024 * 1024
    assert peak < 64 * 1024 * 1024


def test_point_lens_branch_is_explicitly_labeled_and_not_planetary_solver() -> None:
    bundle = SyntheticGenerator(
        small_config(event_type="stellar_microlensing", visits=1, local_steps=13)
    ).generate()

    assert bundle.labels.event_type == "stellar_microlensing"
    assert bundle.labels.microlensing_solver_tier == "point_lens_analytic"
    assert bundle.labels.transit_depth[0] > 0.0
    assert bundle.labels.event_mask is not None
    assert np.max(bundle.injected.wavelength_tokens[..., 1]) > 1.0
