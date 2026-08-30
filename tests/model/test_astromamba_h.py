import pytest
import torch

from model.astromamba_h import (
    AstroMambaH,
    AstroMambaHConfig,
    AstroMambaHInputs,
    ObjectContextEncoder,
)


def tiny_config() -> AstroMambaHConfig:
    return AstroMambaHConfig(
        raster_channels=6,
        wavelength_feature_dim=8,
        object_feature_dim=12,
        geometry_feature_dim=10,
        coverage_feature_dim=6,
        coverage_map_channels=2,
        local_time_feature_dim=5,
        long_time_feature_dim=5,
        stage_channels=(8, 12, 16, 20),
        embedding_dim=16,
        temporal_width=16,
        source_top_k=3,
        context_token_count=2,
        fusion_blocks=1,
        temporal_blocks=1,
        canonical_wavelength_bins=5,
        wavelength_fourier_features=2,
        period_bin_count=7,
        heatmap_rank=3,
    )


def make_inputs(config: AstroMambaHConfig, visits: int = 2, steps: int = 2):
    batch = 1
    wavelength_tokens = 4
    objects = 3
    return AstroMambaHInputs(
        raster=torch.randn(batch, visits, steps, config.raster_channels, 720, 1280),
        wavelength_tokens=torch.randn(
            batch, visits, steps, wavelength_tokens, config.wavelength_feature_dim
        ),
        wavelength_mask=torch.tensor(
            [[[[True, True, True, False]] * steps] * visits], dtype=torch.bool
        ),
        object_tokens=torch.randn(
            batch, visits, objects, config.object_feature_dim
        ),
        object_mask=torch.tensor(
            [[[True, True, False]] * visits], dtype=torch.bool
        ),
        geometry=torch.randn(batch, visits, steps, config.geometry_feature_dim),
        exposure_duration=torch.rand(batch, visits, steps, 1) + 1.0,
        coverage_vector=torch.randn(
            batch, visits, steps, config.coverage_feature_dim
        ),
        coverage_map=torch.randn(
            batch,
            visits,
            steps,
            config.coverage_map_channels,
            90,
            160,
        ),
        local_time=torch.randn(
            batch, visits, steps, config.local_time_feature_dim
        ),
        long_time=torch.randn(batch, visits, config.long_time_feature_dim),
    )


def test_config_preserves_spec_raster_and_multiscale_geometry():
    config = tiny_config()

    assert config.input_size == (720, 1280)
    assert config.spatial_resolutions == (
        (180, 320),
        (180, 320),
        (90, 160),
        (45, 80),
        (23, 40),
    )
    assert config.heatmap_size == (90, 160)


def test_input_contract_rejects_wrong_raster_size_and_keeps_token_masks():
    config = tiny_config()
    inputs = make_inputs(config, visits=1, steps=1)
    inputs.validate(config)

    assert inputs.wavelength_mask.shape == (1, 1, 1, 4)
    assert inputs.wavelength_mask[0, 0, 0].tolist() == [True, True, True, False]

    invalid = make_inputs(config, visits=1, steps=1)
    invalid.raster = invalid.raster[..., :719, :]
    with pytest.raises(ValueError, match="raster.*720x1280"):
        invalid.validate(config)


def test_object_context_pools_over_objects_not_embedding_width():
    config = tiny_config()
    encoder = ObjectContextEncoder(config)
    tokens = torch.randn(2, 3, config.object_feature_dim)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    per_object, regional = encoder(tokens, mask)

    assert per_object.shape == (2, 3, config.embedding_dim)
    assert regional.shape == (2, config.embedding_dim)


def test_forward_exposes_all_architecture_slices_and_dense_heatmaps():
    config = tiny_config()
    model = AstroMambaH(config).eval()
    inputs = make_inputs(config)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs["source_tokens"].shape == (1, 2, 2, 3, 16)
    assert outputs["background_tokens"].shape == (1, 2, 2, 2, 16)
    assert outputs["object_context"].shape == (1, 2, 2, 16)
    assert outputs["geometry_coverage_condition"].shape == (1, 2, 2, 16)
    assert outputs["wavelength_tokens"].shape == (1, 2, 2, 4, 16)
    assert outputs["local_event_tokens"].shape == (1, 2, 2, 16)
    assert outputs["long_time_tokens"].shape == (1, 2, 16)
    assert outputs["period_proposal"]["scores"].shape == (1, 7)
    assert outputs["period_proposal"]["features"].shape == (1, 16)

    assert outputs["heatmaps"].shape == (1, 2, 2, 5, 6, 90, 160)
    assert outputs["source_heatmaps"].shape == (1, 2, 2, 3, 5, 6, 90, 160)
    assert outputs["wavelength_availability"].shape == (1, 2, 2, 5)
    assert outputs["source_heatmap"].shape == (1, 2, 2, 90, 160)
    assert outputs["candidate_heatmap"].shape == (1, 2, 2, 90, 160)
    assert outputs["heatmap_feature_names"] == (
        "normalized_signal",
        "transit_compatible_signal",
        "source_presence",
        "uncertainty",
        "validity_mask",
        "interpolation_mask",
    )

    assert outputs["global_heads"]["candidate_probability"].shape == (1,)
    assert outputs["orbit"]["period_posterior"].shape == (1, 7)
    assert outputs["orbit"]["constraint_statuses"] == (
        "well_constrained",
        "weakly_constrained",
        "prior_dominated",
        "unconstrained",
    )


def test_parameter_count_is_reported_from_real_parameters():
    model = AstroMambaH(tiny_config())

    expected = sum(parameter.numel() for parameter in model.parameters())

    assert model.parameter_count == expected
    assert model.parameter_count > 0
