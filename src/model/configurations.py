"""Named AstroMamba-H capacity presets."""

from __future__ import annotations

from .astromamba_h import AstroMambaHConfig


def research_config() -> AstroMambaHConfig:
    """Return the V2 82--86M-parameter research configuration.

    The default :class:`AstroMambaHConfig` remains intentionally small for
    CPU tests and development. This preset preserves the same input/output
    contract while widening the existing multiscale spatial path, temporal
    representation, and spatial decoder to meet the research budget.
    """

    return AstroMambaHConfig(
        stage_channels=(192, 384, 768, 1024),
        embedding_dim=512,
        temporal_width=768,
        source_top_k=96,
        context_token_count=32,
        fusion_blocks=3,
        fusion_heads=8,
        temporal_blocks=8,
        canonical_wavelength_bins=16,
        wavelength_fourier_features=4,
        heatmap_rank=8,
        period_bin_count=32,
        period_feature_dim=16,
        decoder_width=704,
        decoder_blocks=6,
    )
