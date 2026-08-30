import torch

from model.astromamba_h import AstroMambaH, AstroMambaHInputs, combine_source_conditioned_event_logits
from model.astromamba_h import AstroMambaHConfig


def tiny_config() -> AstroMambaHConfig:
    return AstroMambaHConfig(
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
        decode_heatmaps=False,
    )


def make_inputs(config: AstroMambaHConfig, visits: int, steps: int) -> AstroMambaHInputs:
    return AstroMambaHInputs(
        raster=torch.randn(1, visits, steps, 6, 720, 1280),
        wavelength_tokens=torch.randn(1, visits, steps, 4, 8),
        wavelength_mask=torch.ones(1, visits, steps, 4, dtype=torch.bool),
        object_tokens=torch.randn(1, visits, 3, 12),
        object_mask=torch.ones(1, visits, 3, dtype=torch.bool),
        geometry=torch.randn(1, visits, steps, 10),
        exposure_duration=torch.ones(1, visits, steps, 1),
        coverage_vector=torch.ones(1, visits, steps, 6),
        local_time=torch.randn(1, visits, steps, 5),
        long_time=torch.randn(1, visits, 5),
    )


def test_inputs_expose_explicit_visit_and_step_masks() -> None:
    config = tiny_config()
    inputs = make_inputs(config, visits=2, steps=3)
    inputs.visit_mask = torch.tensor([[True, False]])
    inputs.step_mask = torch.tensor([[[True, True, False], [False, False, False]]])
    inputs.validate(config)
    assert inputs.visit_mask.shape == (1, 2)
    assert inputs.step_mask.shape == (1, 2, 3)


def test_model_keeps_source_identity_and_emits_level_specific_logits() -> None:
    config = tiny_config()
    model = AstroMambaH(config)
    inputs = make_inputs(config, visits=2, steps=3)
    inputs.source_xy = torch.tensor([[0.25, 0.75]])
    outputs = model(inputs)

    assert outputs["source_anchor_indices"].shape == (1, config.source_top_k)
    assert torch.allclose(outputs["source_anchor_xy"][0, 0], torch.tensor([0.25, 0.75]), atol=0.02)
    assert outputs["source_event_logits"].shape == (1, config.source_top_k)
    assert outputs["visit_event_logits"].shape == (1, 2)
    assert outputs["global_event_logits"].shape == (1,)
    assert outputs["source_logits"].shape[:3] == (1, 2, 3)
    assert outputs["head_logits"]["event"].shape == (1,)
    assert outputs["orbit"]["transit_time_offset_by_source"].shape == (1, config.source_top_k)
    assert torch.all((outputs["orbit"]["period_is_constrained_probability_by_source"] >= 0.0) &
                     (outputs["orbit"]["period_is_constrained_probability_by_source"] <= 1.0))


def test_deep_spatial_stages_reach_the_loss() -> None:
    config = tiny_config()
    model = AstroMambaH(config)
    outputs = model(make_inputs(config, visits=1, steps=1))
    outputs["global_event_logits"].sum().backward()

    assert model.spatial_backbone.stage3.block[0].weight.grad is not None
    assert model.spatial_backbone.stage4.block[0].weight.grad is not None


def test_spatial_chunking_supports_temporal_inputs() -> None:
    config = tiny_config()
    config = AstroMambaHConfig(**{**config.__dict__, "spatial_chunk_size": 1})
    model = AstroMambaH(config)
    outputs = model(make_inputs(config, visits=1, steps=2))

    assert outputs["local_source_tokens"].shape[2] == 2


def test_empty_modalities_have_zero_contribution_not_attention_bias() -> None:
    config = tiny_config()
    inputs = make_inputs(config, visits=1, steps=1)
    inputs.wavelength_mask.zero_()
    inputs.object_mask.zero_()
    outputs = AstroMambaH(config)(inputs)
    assert torch.allclose(outputs["missing_modality_flags"], torch.ones(1, 2))


def test_global_event_logit_uses_source_conditioned_evidence() -> None:
    pooled_backbone = torch.tensor([4.0])
    source_event = torch.tensor([[-4.0, -5.0]])
    source_photometry = torch.tensor([[-1.0]])

    result = combine_source_conditioned_event_logits(
        pooled_backbone,
        source_event,
        source_photometry,
    )

    assert result.shape == (1,)
    assert result.item() < 0.0
