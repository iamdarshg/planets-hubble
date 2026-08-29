"""Named AstroMamba-H capacity presets."""

from __future__ import annotations

from .astromamba_h import AstroMambaHConfig


def research_config() -> AstroMambaHConfig:
    """Return the measured 70--90M-parameter research configuration.

    The default :class:`AstroMambaHConfig` remains intentionally small for
    CPU tests and development.  This preset preserves the same input/output
    contract while increasing spatial, fusion, and temporal capacity for the
    intended local-GPU training experiments.
    """

    return AstroMambaHConfig(
        stage_channels=(128, 256, 512, 768),
        embedding_dim=512,
        temporal_width=512,
        source_top_k=96,
        context_token_count=32,
        fusion_blocks=12,
        fusion_heads=8,
        temporal_blocks=24,
        canonical_wavelength_bins=16,
        wavelength_fourier_features=4,
        heatmap_rank=8,
        period_bin_count=32,
        period_feature_dim=16,
    )
