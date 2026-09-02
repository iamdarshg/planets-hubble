"""A testable PyTorch architecture scaffold for the AstroMamba-H design.

This module deliberately stops at an architecture boundary.  It does not train,
download data, perform WCS transforms, or make scientific claims.  Variable
length wavelength and object collections are represented by padded tensors plus
explicit masks so that missing data remains distinguishable from a zero value.

The temporal mixers are small gated convolutional mixers implemented with core
PyTorch.  They stand in for the proposed Mamba-2 stages until a separately
validated Mamba dependency and sequence benchmark are selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


HEATMAP_FEATURE_NAMES = (
    "normalized_signal",
    "transit_compatible_signal",
    "source_presence",
    "uncertainty",
    "validity_mask",
    "interpolation_mask",
)
ORBIT_CONSTRAINT_STATUSES = (
    "well_constrained",
    "weakly_constrained",
    "prior_dominated",
    "unconstrained",
)


def combine_source_conditioned_event_logits(
    pooled_backbone_logits: Tensor,
    source_event_logits: Tensor,
    source_photometry_event_logits: Tensor,
    *,
    backbone_weight: float = 0.5,
    photometry_weight: float = 0.5,
) -> Tensor:
    """Combine patch and anchor evidence without losing source identity.

    The pooled backbone is useful for discovery, but it can be driven by an
    unrelated bright or unstable source in the patch.  The known/persistent
    source anchor therefore supplies the primary event evidence.  Keeping a
    bounded backbone contribution preserves a discovery-mode gradient while
    preventing a contradictory patch-level false positive from overriding the
    source-conditioned signal.
    """

    if pooled_backbone_logits.ndim != 1:
        raise ValueError("pooled_backbone_logits must have shape [batch]")
    if source_event_logits.ndim != 2 or source_event_logits.shape[0] != pooled_backbone_logits.shape[0]:
        raise ValueError("source_event_logits must have shape [batch, source]")
    if source_photometry_event_logits.ndim != 2:
        raise ValueError("source_photometry_event_logits must have shape [batch, 1]")
    if source_photometry_event_logits.shape != (pooled_backbone_logits.shape[0], 1):
        raise ValueError("source_photometry_event_logits must have shape [batch, 1]")
    if not 0.0 <= backbone_weight <= 1.0 or not 0.0 <= photometry_weight <= 1.0:
        raise ValueError("evidence weights must be in [0, 1]")
    return (
        source_event_logits[:, 0]
        + backbone_weight * pooled_backbone_logits
        + photometry_weight * source_photometry_event_logits[:, 0]
    )


@dataclass(frozen=True)
class AstroMambaHConfig:
    """Shape and capacity contract for the scaffold.

    The defaults are intentionally smaller than the specification's eventual
    82--86M research target.  They provide the same factorized interfaces while
    keeping construction and shape tests practical on a CPU-only machine.
    Increase the dimensions only after data and training gates are defined.
    """

    input_height: int = 720
    input_width: int = 1280
    heatmap_stride: int = 8
    raster_channels: int = 6
    wavelength_feature_dim: int = 8
    object_feature_dim: int = 12
    geometry_feature_dim: int = 10
    coverage_feature_dim: int = 6
    coverage_map_channels: int = 2
    local_time_feature_dim: int = 5
    long_time_feature_dim: int = 5
    stage_channels: Tuple[int, int, int, int] = (32, 48, 64, 96)
    embedding_dim: int = 64
    temporal_width: int = 64
    source_top_k: int = 8
    context_token_count: int = 4
    fusion_blocks: int = 2
    fusion_heads: int = 4
    temporal_blocks: int = 2
    canonical_wavelength_bins: int = 16
    wavelength_fourier_features: int = 4
    heatmap_rank: int = 8
    heatmap_features: int = 6
    period_bin_count: int = 32
    period_feature_dim: int = 16
    temporal_backend: str = "auto"
    mamba_state: int = 64
    mamba_conv: int = 4
    mamba_expand: int = 2
    wavelength_set_blocks: int = 2
    object_set_blocks: int = 2
    decoder_width: int = 512
    decoder_blocks: int = 3
    # Number of raster frames retained in one spatial activation graph.
    # Values smaller than the flattened frame count use activation
    # checkpointing without changing source-token or temporal dimensions.
    spatial_chunk_size: int = 1
    # Dense wavelength heatmaps are required for inference, but are optional
    # during memory-constrained sequence training when the structured heads
    # are the active objective.
    decode_heatmaps: bool = True
    # Kepler target-pixel files are compact detector cutouts.  This explicit
    # mode keeps their native pixels (the data adapter pads only) while
    # preserving the default HST 720x1280 contract.
    allow_compact_input: bool = False

    def __post_init__(self) -> None:
        if self.allow_compact_input:
            if self.input_height < 32 or self.input_width < 32:
                raise ValueError("compact AstroMamba-H inputs must be at least 32x32")
            if self.input_height % 32 or self.input_width % 32:
                raise ValueError("compact AstroMamba-H inputs must be divisible by 32")
        elif (self.input_height, self.input_width) != (720, 1280):
            raise ValueError("AstroMamba-H requires a 720x1280 input raster unless compact mode is enabled")
        if self.heatmap_stride != 8:
            raise ValueError("the scaffold currently exposes an /8 heatmap grid")
        if len(self.stage_channels) != 4 or any(c < 1 for c in self.stage_channels):
            raise ValueError("stage_channels must contain four positive widths")
        positive = (
            self.raster_channels,
            self.wavelength_feature_dim,
            self.object_feature_dim,
            self.geometry_feature_dim,
            self.coverage_feature_dim,
            self.coverage_map_channels,
            self.local_time_feature_dim,
            self.long_time_feature_dim,
            self.embedding_dim,
            self.temporal_width,
            self.source_top_k,
            self.context_token_count,
            self.fusion_blocks,
            self.fusion_heads,
            self.temporal_blocks,
            self.canonical_wavelength_bins,
            self.wavelength_fourier_features,
            self.heatmap_rank,
            self.period_bin_count,
            self.period_feature_dim,
        )
        if any(value < 1 for value in positive):
            raise ValueError("model dimensions must be positive")
        if self.raster_channels < 5:
            raise ValueError("raster_channels must include validity and interpolation masks")
        if self.heatmap_features != len(HEATMAP_FEATURE_NAMES):
            raise ValueError("heatmap_features must expose the six required feature channels")
        if self.embedding_dim % self.fusion_heads != 0:
            raise ValueError("embedding_dim must be divisible by fusion_heads")
        if self.temporal_backend not in {"auto", "gated_conv", "mamba2"}:
            raise ValueError("temporal_backend must be auto, gated_conv, or mamba2")
        if min(
            self.mamba_state,
            self.mamba_conv,
            self.mamba_expand,
            self.wavelength_set_blocks,
            self.object_set_blocks,
            self.decoder_width,
            self.decoder_blocks,
            self.spatial_chunk_size,
        ) < 1:
            raise ValueError("Mamba dimensions must be positive")

    @property
    def input_size(self) -> Tuple[int, int]:
        return (self.input_height, self.input_width)

    @property
    def spatial_resolutions(self) -> Tuple[Tuple[int, int], ...]:
        """Feature resolutions in (height, width), matching SPEC.md."""

        height, width = self.input_size
        return (
            (height // 4, width // 4),
            (height // 4, width // 4),
            (height // 8, width // 8),
            (height // 16, width // 16),
            ((height + 31) // 32, (width + 31) // 32),
        )

    @property
    def heatmap_size(self) -> Tuple[int, int]:
        return (self.input_height // self.heatmap_stride, self.input_width // self.heatmap_stride)


@dataclass
class AstroMambaHInputs:
    """Padded tensor contract for one batch of irregular observations.

    Shapes use ``B`` batch, ``V`` long-time visits, ``S`` local steps, ``W``
    wavelength tokens, and ``O`` regional objects.  The first wavelength
    feature is a normalized logarithmic wavelength/energy coordinate.  The
    remaining features are measurement value, uncertainty, bandwidth,
    exposure/response information, and other source-independent measurements.
    """

    raster: Tensor
    wavelength_tokens: Tensor
    wavelength_mask: Tensor
    object_tokens: Tensor
    object_mask: Tensor
    geometry: Tensor
    exposure_duration: Tensor
    coverage_vector: Tensor
    local_time: Tensor
    long_time: Tensor
    coverage_map: Optional[Tensor] = None
    visit_mask: Optional[Tensor] = None
    step_mask: Optional[Tensor] = None
    source_xy: Optional[Tensor] = None

    def validate(self, config: AstroMambaHConfig) -> None:
        if self.raster.ndim != 6:
            raise ValueError("raster must have shape [B,V,S,C,720,1280]")
        b, visits, steps, channels, height, width = self.raster.shape
        if (height, width) != config.input_size:
            raise ValueError(f"raster must have spatial size {config.input_height}x{config.input_width}")
        if channels != config.raster_channels:
            raise ValueError(f"raster has {channels} channels, expected {config.raster_channels}")

        self._require_shape(
            self.wavelength_tokens,
            (b, visits, steps, None, config.wavelength_feature_dim),
            "wavelength_tokens",
        )
        wavelength_count = self.wavelength_tokens.shape[3]
        self._require_shape(
            self.wavelength_mask,
            (b, visits, steps, wavelength_count),
            "wavelength_mask",
        )
        if self.wavelength_mask.dtype != torch.bool:
            raise ValueError("wavelength_mask must use bool dtype")

        self._require_shape(
            self.object_tokens,
            (b, visits, None, config.object_feature_dim),
            "object_tokens",
        )
        object_count = self.object_tokens.shape[2]
        self._require_shape(self.object_mask, (b, visits, object_count), "object_mask")
        if self.object_mask.dtype != torch.bool:
            raise ValueError("object_mask must use bool dtype")

        self._require_shape(
            self.geometry,
            (b, visits, steps, config.geometry_feature_dim),
            "geometry",
        )
        self._require_shape(self.exposure_duration, (b, visits, steps, 1), "exposure_duration")
        self._require_shape(
            self.coverage_vector,
            (b, visits, steps, config.coverage_feature_dim),
            "coverage_vector",
        )
        self._require_shape(
            self.local_time,
            (b, visits, steps, config.local_time_feature_dim),
            "local_time",
        )
        self._require_shape(self.long_time, (b, visits, config.long_time_feature_dim), "long_time")

        if self.visit_mask is not None:
            self._require_shape(self.visit_mask, (b, visits), "visit_mask")
            if self.visit_mask.dtype != torch.bool:
                raise ValueError("visit_mask must use bool dtype")
        if self.step_mask is not None:
            self._require_shape(self.step_mask, (b, visits, steps), "step_mask")
            if self.step_mask.dtype != torch.bool:
                raise ValueError("step_mask must use bool dtype")

        if self.source_xy is not None:
            self._require_shape(self.source_xy, (b, 2), "source_xy")
            if not torch.isfinite(self.source_xy).all():
                raise ValueError("source_xy must be finite")
            if torch.any((self.source_xy < 0.0) | (self.source_xy > 1.0)):
                raise ValueError("source_xy must be normalized to [0, 1]")

        if self.coverage_map is not None:
            self._require_shape(
                self.coverage_map,
                (b, visits, steps, config.coverage_map_channels, *config.heatmap_size),
                "coverage_map",
            )

    @staticmethod
    def _require_shape(value: Tensor, expected: Tuple[Optional[int], ...], name: str) -> None:
        if value.ndim != len(expected):
            raise ValueError(f"{name} has rank {value.ndim}, expected {len(expected)}")
        for index, (actual, wanted) in enumerate(zip(value.shape, expected)):
            if wanted is not None and actual != wanted:
                raise ValueError(f"{name} has invalid dimension {index}: expected {wanted}, got {actual}")


class SpatialStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.block(value)


class SpatialBackbone(nn.Module):
    def __init__(self, config: AstroMambaHConfig) -> None:
        super().__init__()
        c0, c1, c2, c3 = config.stage_channels
        self.stem = nn.Sequential(
            nn.Conv2d(config.raster_channels, c0, kernel_size=4, stride=4),
            nn.GroupNorm(1, c0),
            nn.GELU(),
        )
        self.stage1 = SpatialStage(c0, c0, stride=1)
        self.stage2 = SpatialStage(c0, c1, stride=2)
        self.stage3 = SpatialStage(c1, c2, stride=2)
        self.stage4 = SpatialStage(c2, c3, stride=2)

    def forward(self, raster: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        stem = self.stem(raster)
        stage1 = self.stage1(stem)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        return stem, stage1, stage2, stage3, stage4


class SpatialFPN(nn.Module):
    """Fuse every backbone stage at the /8 source-token resolution."""

    def __init__(self, config: AstroMambaHConfig) -> None:
        super().__init__()
        c0, c1, c2, c3 = config.stage_channels
        self.stage1 = nn.Conv2d(c0, c1, kernel_size=1, stride=2)
        self.stage2 = nn.Conv2d(c1, c1, kernel_size=1)
        self.stage3 = nn.Conv2d(c2, c1, kernel_size=1)
        self.stage4 = nn.Conv2d(c3, c1, kernel_size=1)
        self.merge = nn.Sequential(
            nn.Conv2d(c1 * 4, c1, kernel_size=3, padding=1),
            nn.GroupNorm(1, c1),
            nn.GELU(),
        )

    def forward(self, stage1: Tensor, stage2: Tensor, stage3: Tensor, stage4: Tensor) -> Tensor:
        target_size = stage2.shape[-2:]
        features = (
            self.stage1(stage1),
            self.stage2(stage2),
            F.interpolate(self.stage3(stage3), size=target_size, mode="bilinear", align_corners=False),
            F.interpolate(self.stage4(stage4), size=target_size, mode="bilinear", align_corners=False),
        )
        return self.merge(torch.cat(features, dim=1))


class SourceTokenizer(nn.Module):
    def __init__(self, channels: int, config: AstroMambaHConfig) -> None:
        super().__init__()
        self.source_projection = nn.Conv2d(channels, config.embedding_dim, kernel_size=1)
        self.background_projection = nn.Linear(channels, config.embedding_dim)
        self.source_score = nn.Conv2d(channels, 1, kernel_size=1)
        self.top_k = config.source_top_k
        self.context_count = config.context_token_count

    def forward(self, feature_map: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, _, height, width = feature_map.shape
        source_logits_map = self.source_score(feature_map)
        flat_scores = source_logits_map.flatten(1)
        top_scores, top_indices = torch.topk(flat_scores, k=self.top_k, dim=1)

        projected = self.source_projection(feature_map).flatten(2).transpose(1, 2)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
        source_tokens = torch.gather(projected, 1, gather_index)

        pooled = F.adaptive_avg_pool2d(feature_map, (self.context_count, 1)).squeeze(-1)
        background_tokens = self.background_projection(pooled.transpose(1, 2))
        return source_tokens, background_tokens, top_scores, source_logits_map.squeeze(1)

    def persistent_forward(
        self, feature_map: Tensor, step_mask: Tensor, source_xy: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Anchor sources once and sample those same locations in every frame."""

        batch, visits, steps, channels, height, width = feature_map.shape
        flat_feature_map = feature_map.reshape(batch, visits * steps, channels, height, width)
        flat_step_mask = step_mask.reshape(batch, visits * steps)
        reference_indices = flat_step_mask.to(dtype=torch.long).argmax(dim=1)
        batch_indices = torch.arange(batch, device=feature_map.device)
        reference = flat_feature_map[batch_indices, reference_indices]
        reference_logits = self.source_score(reference).flatten(1)
        top_scores, anchor_indices = torch.topk(reference_logits, k=self.top_k, dim=1)
        if source_xy is not None:
            # A patch may be deliberately uncentered.  Preserve one known
            # catalog/source anchor while retaining learned top-k proposals
            # for discovery of additional sources.
            source_x = torch.round(source_xy[:, 0].clamp(0.0, 1.0) * (width - 1)).to(torch.long)
            source_y = torch.round(source_xy[:, 1].clamp(0.0, 1.0) * (height - 1)).to(torch.long)
            hinted_index = source_y * width + source_x
            anchor_indices = anchor_indices.clone()
            anchor_indices[:, 0] = hinted_index
            top_scores = torch.gather(reference_logits, 1, anchor_indices)
        flattened = feature_map.reshape(batch * visits * steps, channels, height, width)
        projected = self.source_projection(flattened).flatten(2).transpose(1, 2)
        repeated_indices = anchor_indices[:, None, None].expand(batch, visits, steps, -1)
        repeated_indices = repeated_indices.reshape(batch * visits * steps, self.top_k)
        gather_index = repeated_indices.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
        tokens = torch.gather(projected, 1, gather_index).reshape(
            batch, visits, steps, self.top_k, projected.shape[-1]
        )

        score_maps = self.source_score(flattened).flatten(1)
        source_scores = torch.gather(score_maps, 1, repeated_indices).reshape(
            batch, visits, steps, self.top_k
        )
        source_logits_map = score_maps.reshape(batch, visits, steps, height, width)
        pooled = F.adaptive_avg_pool2d(flattened, (self.context_count, 1)).squeeze(-1)
        background = self.background_projection(pooled.transpose(1, 2)).reshape(
            batch, visits, steps, self.context_count, -1
        )
        # Keep invalid frames numerically inert while preserving their explicit
        # positions for downstream diagnostics and temporal masks.
        tokens = tokens * step_mask[..., None, None].to(tokens.dtype)
        source_scores = source_scores * step_mask[..., None].to(source_scores.dtype)
        background = background * step_mask[..., None, None].to(background.dtype)
        return tokens, background, source_scores, source_logits_map, anchor_indices


class MaskedSetBlock(nn.Module):
    """Permutation-equivariant attention block for variable-length modalities."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        present = mask.any(dim=1)
        safe_mask = mask.clone()
        empty = ~present
        if empty.any():
            safe_mask[empty, 0] = True
        normalized = self.norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_mask,
        )
        result = tokens + attended + self.feed_forward(tokens)
        return result * mask.unsqueeze(-1).to(result.dtype)


class WavelengthEncoder(nn.Module):
    def __init__(self, config: AstroMambaHConfig) -> None:
        super().__init__()
        self.fourier_features = config.wavelength_fourier_features
        self.canonical_bins = config.canonical_wavelength_bins
        frequencies = 2.0 ** torch.arange(self.fourier_features, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        input_width = config.wavelength_feature_dim + 2 * self.fourier_features
        self.projection = nn.Sequential(
            nn.Linear(input_width, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.set_blocks = nn.ModuleList(
            MaskedSetBlock(config.embedding_dim, config.fusion_heads)
            for _ in range(config.wavelength_set_blocks)
        )

    def forward(self, tokens: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        coordinate = tokens[..., :1]
        argument = coordinate * self.frequencies.view(1, 1, -1) * torch.pi
        encoded = torch.cat((tokens, argument.sin(), argument.cos()), dim=-1)
        embeddings = self.projection(encoded)
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        embeddings = embeddings * weights
        for block in self.set_blocks:
            embeddings = block(embeddings, mask)
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = embeddings.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return embeddings, pooled, self.availability(tokens[..., 0], mask, self.canonical_bins)

    @staticmethod
    def availability(coordinate: Tensor, mask: Tensor, bins: int = 16) -> Tensor:
        grid = torch.linspace(0.0, 1.0, bins, device=coordinate.device, dtype=coordinate.dtype)
        distance = coordinate.unsqueeze(-1) - grid.view(1, 1, -1)
        weights = torch.exp(-0.5 * (distance / 0.15).square())
        weights = weights * mask.unsqueeze(-1).to(weights.dtype)
        return weights.sum(dim=1).clamp(0.0, 1.0)


class ObjectContextEncoder(nn.Module):
    def __init__(self, config: AstroMambaHConfig) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.object_feature_dim, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.norm = nn.LayerNorm(config.embedding_dim)
        self.attention = nn.MultiheadAttention(
            config.embedding_dim, config.fusion_heads, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
        )
        self.set_blocks = nn.ModuleList(
            MaskedSetBlock(config.embedding_dim, config.fusion_heads)
            for _ in range(config.object_set_blocks)
        )

    def forward(self, tokens: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        embeddings = self.projection(tokens)
        present = mask.any(dim=1)
        safe_mask = mask.clone()
        empty = ~present
        if empty.any():
            safe_mask[empty, 0] = True
        attended, _ = self.attention(
            self.norm(embeddings),
            self.norm(embeddings),
            self.norm(embeddings),
            key_padding_mask=~safe_mask,
        )
        embeddings = embeddings + attended + self.feed_forward(self.norm(embeddings))
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        embeddings = embeddings * weights
        for block in self.set_blocks:
            embeddings = block(embeddings, mask)
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        regional = embeddings.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return embeddings, regional


class CrossModalBlock(nn.Module):
    def __init__(self, config: AstroMambaHConfig) -> None:
        super().__init__()
        dim = config.embedding_dim
        self.source_norm = nn.LayerNorm(dim)
        self.wave_norm = nn.LayerNorm(dim)
        self.object_norm = nn.LayerNorm(dim)
        self.wave_attention = nn.MultiheadAttention(dim, config.fusion_heads, batch_first=True)
        self.object_attention = nn.MultiheadAttention(dim, config.fusion_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    @staticmethod
    def _safe_mask(mask: Tensor) -> Tensor:
        safe = mask.clone()
        empty = ~safe.any(dim=1)
        if empty.any():
            safe[empty, 0] = True
        return ~safe

    def forward(
        self,
        source: Tensor,
        wavelength: Tensor,
        wavelength_mask: Tensor,
        objects: Tensor,
        object_mask: Tensor,
    ) -> Tensor:
        query = self.source_norm(source)
        wave = self.wave_norm(wavelength)
        safe_wave = wave * wavelength_mask.unsqueeze(-1).to(wave.dtype)
        wave_present = wavelength_mask.any(dim=1)
        wave_result, _ = self.wave_attention(
            query, safe_wave, safe_wave, key_padding_mask=self._safe_mask(wavelength_mask)
        )
        source = source + wave_result * wave_present.unsqueeze(1).unsqueeze(-1).to(wave_result.dtype)

        query = self.source_norm(source)
        object_values = self.object_norm(objects)
        safe_objects = object_values * object_mask.unsqueeze(-1).to(object_values.dtype)
        object_present = object_mask.any(dim=1)
        object_result, _ = self.object_attention(
            query, safe_objects, safe_objects, key_padding_mask=self._safe_mask(object_mask)
        )
        return source + object_result * object_present.unsqueeze(1).unsqueeze(-1).to(object_result.dtype) + self.feed_forward(source)


class GatedTemporalBlock(nn.Module):
    """Portable gated-convolution fallback for tests and unsupported platforms."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.in_projection = nn.Linear(width, width * 2)
        self.depthwise = nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width)
        self.out_projection = nn.Linear(width, width)

    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            sequence = sequence * mask.unsqueeze(-1).to(sequence.dtype)
        normalized = self.norm(sequence)
        value, gate = self.in_projection(normalized).chunk(2, dim=-1)
        value = self.depthwise(value.transpose(1, 2)).transpose(1, 2)
        mixed = F.silu(value) * torch.sigmoid(gate)
        result = sequence + self.out_projection(mixed)
        return result if mask is None else result * mask.unsqueeze(-1).to(result.dtype)


class Mamba2TemporalBlock(nn.Module):
    """Optional real Mamba-2 block, loaded only when mamba-ssm is installed."""

    def __init__(self, width: int, *, state: int, conv: int, expand: int) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:
            raise RuntimeError(
                "temporal_backend='mamba2' requires the optional mamba-ssm package"
            ) from exc
        self.norm = nn.LayerNorm(width)
        self.mixer = Mamba2(d_model=width, d_state=state, d_conv=conv, expand=expand)

    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None:
            sequence = sequence * mask.unsqueeze(-1).to(sequence.dtype)
        result = sequence + self.mixer(self.norm(sequence))
        return result if mask is None else result * mask.unsqueeze(-1).to(result.dtype)


class TemporalStage(nn.Module):
    def __init__(self, width: int, blocks: int, config: AstroMambaHConfig) -> None:
        super().__init__()
        backend = config.temporal_backend
        if backend == "auto":
            try:
                from mamba_ssm import Mamba2  # noqa: F401
                backend = "mamba2"
            except ImportError:
                backend = "gated_conv"
        self.backend = backend
        block_type = Mamba2TemporalBlock if backend == "mamba2" else GatedTemporalBlock
        self.blocks = nn.ModuleList(
            block_type(width, state=config.mamba_state, conv=config.mamba_conv, expand=config.mamba_expand)
            if backend == "mamba2"
            else block_type(width)
            for _ in range(blocks)
        )

    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            sequence = block(sequence, mask=mask)
        return sequence


class SpatialDecoderBlock(nn.Module):
    """Parameter-efficient ConvNeXt-like refinement at the /8 FPN scale."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.depthwise = nn.Conv2d(width, width, kernel_size=7, padding=3, groups=width)
        self.pointwise = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, kernel_size=1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.pointwise(self.depthwise(self.norm(value)))


class AstroMambaH(nn.Module):
    """Factorized AstroMamba-H model-building scaffold.

    The returned dictionary intentionally keeps intermediate representations
    inspectable: source/background tokens, native wavelength tokens, local and
    long-time tokens, period proposals, heatmaps, and global quality heads.
    """

    def __init__(self, config: Optional[AstroMambaHConfig] = None) -> None:
        super().__init__()
        self.config = config or AstroMambaHConfig()
        config = self.config
        c0, c1, c2, c3 = config.stage_channels

        self.spatial_backbone = SpatialBackbone(config)
        self.spatial_fpn = SpatialFPN(config)
        self.source_tokenizer = SourceTokenizer(c1, config)
        self.wavelength_encoder = WavelengthEncoder(config)
        self.object_context_encoder = ObjectContextEncoder(config)

        condition_width = (
            config.geometry_feature_dim
            + 1
            + config.coverage_feature_dim
            + config.coverage_map_channels
        )
        self.geometry_coverage_encoder = nn.Sequential(
            nn.Linear(condition_width, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.fusion_projection = nn.Linear(config.embedding_dim, config.temporal_width)
        self.source_photometry_projection = nn.Sequential(
            nn.Linear(2, config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )
        self.source_photometry_event = nn.Linear(2, 1)
        self.cross_modal_fusion = nn.ModuleList(
            CrossModalBlock(config) for _ in range(config.fusion_blocks)
        )

        self.local_time_projection = nn.Linear(config.local_time_feature_dim, config.temporal_width)
        self.local_input_projection = nn.Linear(config.embedding_dim, config.temporal_width)
        self.local_temporal = TemporalStage(config.temporal_width, config.temporal_blocks, config)
        self.long_time_projection = nn.Linear(config.long_time_feature_dim, config.temporal_width)
        self.long_temporal = TemporalStage(config.temporal_width, config.temporal_blocks, config)
        self.source_pool_score = nn.Linear(config.temporal_width, 1)

        self.period_proposal = nn.ModuleDict(
            {
                "features": nn.Sequential(
                    nn.Linear(config.temporal_width, config.period_feature_dim),
                    nn.GELU(),
                    nn.Linear(config.period_feature_dim, config.period_feature_dim),
                ),
                "scores": nn.Linear(config.temporal_width, config.period_bin_count),
                "constraint": nn.Linear(config.temporal_width, len(ORBIT_CONSTRAINT_STATUSES)),
            }
        )

        if config.decode_heatmaps:
            self.spatial_temporal_decoder = nn.ModuleDict(
                {
                    "feature_projection": nn.Conv2d(
                        c1, config.decoder_width, kernel_size=1
                    ),
                    "refinement": nn.ModuleList(
                        SpatialDecoderBlock(config.decoder_width)
                        for _ in range(config.decoder_blocks)
                    ),
                    "spatial_factor": nn.Conv2d(
                        config.decoder_width, config.heatmap_rank, kernel_size=1
                    ),
                    "temporal_factor": nn.Linear(
                        config.temporal_width,
                        config.canonical_wavelength_bins
                        * config.heatmap_features
                        * config.heatmap_rank,
                    ),
                }
            )
        else:
            # Training-only sequence workers can omit the dense decoder
            # entirely; its parameters are not involved in structured losses.
            self.spatial_temporal_decoder = nn.ModuleDict()
        self.prediction_heads = nn.ModuleDict(
            {
                "candidate": nn.Linear(config.temporal_width, 1),
                "event": nn.Linear(config.temporal_width, 1),
                "source_event": nn.Linear(config.temporal_width, 1),
                "visit_event": nn.Linear(config.temporal_width, 1),
                "global_event": nn.Linear(config.temporal_width, 1),
                "artifact": nn.Linear(config.temporal_width, 1),
                "ood": nn.Linear(config.temporal_width, 1),
                "coverage": nn.Linear(config.temporal_width, 1),
                "sufficiency": nn.Linear(config.temporal_width, 1),
                "frame_event": nn.Linear(config.temporal_width, 1),
            }
        )
        period_grid = torch.logspace(-1, 4, config.period_bin_count)
        self.register_buffer("period_grid_days", period_grid, persistent=False)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_count_by_component(self) -> Dict[str, int]:
        components = {
            "spatial_backbone": nn.ModuleList([self.spatial_backbone, self.spatial_fpn]),
            "source_tokenizer": self.source_tokenizer,
            "wavelength_encoder": self.wavelength_encoder,
            "source_photometry": nn.ModuleList(
                [self.source_photometry_projection, self.source_photometry_event]
            ),
            "object_context_encoder": self.object_context_encoder,
            "geometry_coverage_encoder": self.geometry_coverage_encoder,
            "cross_modal_fusion": self.cross_modal_fusion,
            "local_temporal": nn.ModuleList(
                [self.local_time_projection, self.local_input_projection, self.local_temporal]
            ),
            "long_temporal": nn.ModuleList([self.long_time_projection, self.long_temporal]),
            "period_proposal": self.period_proposal,
            "spatial_temporal_decoder": self.spatial_temporal_decoder,
            "prediction_heads": nn.ModuleList([self.fusion_projection, self.prediction_heads]),
        }
        return {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in components.items()
        }

    def _spatial_features(self, raster: Tensor) -> Tensor:
        """Extract an /8 FPN map for one raster chunk."""

        _, stage1, stage2, stage3, stage4 = self.spatial_backbone(raster)
        return self.spatial_fpn(stage1, stage2, stage3, stage4)

    def forward(self, inputs: AstroMambaHInputs) -> Dict[str, object]:
        inputs.validate(self.config)
        config = self.config
        batch, visits, steps, _, _, _ = inputs.raster.shape
        frame_count = batch * visits * steps

        visit_mask = inputs.visit_mask
        if visit_mask is None:
            visit_mask = torch.ones(batch, visits, dtype=torch.bool, device=inputs.raster.device)
        step_mask = inputs.step_mask
        if step_mask is None:
            step_mask = torch.ones(batch, visits, steps, dtype=torch.bool, device=inputs.raster.device)
        step_mask = step_mask & visit_mask.unsqueeze(-1)

        raster = inputs.raster.reshape(frame_count, config.raster_channels, *config.input_size)
        chunk_size = min(self.config.spatial_chunk_size, frame_count)
        source_map_chunks = []
        for start in range(0, frame_count, chunk_size):
            raster_chunk = raster[start:start + chunk_size]
            if self.training and torch.is_grad_enabled() and frame_count > chunk_size:
                source_map_chunks.append(
                    checkpoint(self._spatial_features, raster_chunk, use_reentrant=False)
                )
            else:
                source_map_chunks.append(self._spatial_features(raster_chunk))
        source_map = torch.cat(source_map_chunks, dim=0)
        source_map_sequence = source_map.reshape(
            batch, visits, steps, source_map.shape[1], source_map.shape[2], source_map.shape[3]
        )
        source_tokens, background_tokens, source_scores, source_logits_map, anchor_indices = (
            self.source_tokenizer.persistent_forward(source_map_sequence, step_mask, inputs.source_xy)
        )

        wavelength = inputs.wavelength_tokens.reshape(
            frame_count, inputs.wavelength_tokens.shape[3], config.wavelength_feature_dim
        )
        wavelength_mask = inputs.wavelength_mask.reshape(frame_count, -1)
        wavelength_embeddings, wavelength_context, availability = self.wavelength_encoder(
            wavelength, wavelength_mask
        )

        objects = inputs.object_tokens[:, :, None].expand(-1, -1, steps, -1, -1)
        object_mask = inputs.object_mask[:, :, None].expand(-1, -1, steps, -1)
        objects = objects.reshape(frame_count, objects.shape[3], config.object_feature_dim)
        object_mask = object_mask.reshape(frame_count, -1)
        object_embeddings, object_context = self.object_context_encoder(objects, object_mask)

        geometry = inputs.geometry.reshape(frame_count, config.geometry_feature_dim)
        exposure = inputs.exposure_duration.reshape(frame_count, 1).clamp_min(1e-8).log()
        coverage = inputs.coverage_vector.reshape(frame_count, config.coverage_feature_dim)
        if inputs.coverage_map is None:
            coverage_map_summary = torch.zeros(
                frame_count,
                config.coverage_map_channels,
                device=raster.device,
                dtype=raster.dtype,
            )
        else:
            coverage_map_summary = inputs.coverage_map.reshape(
                frame_count, config.coverage_map_channels, *config.heatmap_size
            ).mean(dim=(-2, -1))
        condition = self.geometry_coverage_encoder(
            torch.cat((geometry, exposure, coverage, coverage_map_summary), dim=-1)
        )

        source_tokens = source_tokens.reshape(frame_count, config.source_top_k, config.embedding_dim)
        source_tokens = source_tokens + condition.unsqueeze(1) + wavelength_context.unsqueeze(1)
        source_tokens = source_tokens + object_context.unsqueeze(1)
        wavelength_weights = inputs.wavelength_mask.unsqueeze(-1).to(inputs.wavelength_tokens.dtype)
        wavelength_present = inputs.wavelength_mask.any(dim=3)
        frame_mask = step_mask.reshape(frame_count)
        # Feed the source-conditioned photometry branch the ratio-like flux
        # and the robust cadence-level temporal score. The latter is already
        # computed from the observed sequence by each data adapter and keeps
        # a short transit from being hidden behind a raw uncertainty scale.
        source_photometry_values = torch.stack(
            (inputs.wavelength_tokens[..., 1], inputs.wavelength_tokens[..., 6]),
            dim=-1,
        )
        source_photometry_values = (
            (source_photometry_values * wavelength_weights).sum(dim=3)
            / wavelength_weights.sum(dim=3).clamp_min(1.0)
        )
        source_photometry = self.source_photometry_projection(source_photometry_values)
        source_photometry = source_photometry * (wavelength_present & step_mask).unsqueeze(-1).to(
            source_photometry.dtype
        )
        source_tokens = source_tokens + source_photometry.reshape(
            frame_count, 1, config.embedding_dim
        )
        for block in self.cross_modal_fusion:
            source_tokens = block(
                source_tokens,
                wavelength_embeddings,
                wavelength_mask,
                object_embeddings,
                object_mask,
            )
        source_tokens = source_tokens * frame_mask[:, None, None].to(source_tokens.dtype)

        source_tokens_by_frame = source_tokens.reshape(batch, visits, steps, config.source_top_k, -1)
        local_time = inputs.local_time
        local_sequence = source_tokens_by_frame.permute(0, 1, 3, 2, 4).reshape(
            batch * visits * config.source_top_k, steps, config.embedding_dim
        )
        local_sequence = self.local_input_projection(local_sequence)
        local_time_embedding = self.local_time_projection(local_time)
        local_time_embedding = local_time_embedding[:, :, None].expand(
            -1, -1, config.source_top_k, -1, -1
        )
        local_time_embedding = local_time_embedding.permute(0, 1, 2, 3, 4).reshape(
            batch * visits * config.source_top_k, steps, config.temporal_width
        )
        local_step_mask = step_mask[:, :, None, :].expand(-1, -1, config.source_top_k, -1).reshape(
            batch * visits * config.source_top_k, steps
        )
        local_sequence = self.local_temporal(
            local_sequence + local_time_embedding, mask=local_step_mask
        )
        local_source_tokens = local_sequence.reshape(
            batch, visits, config.source_top_k, steps, config.temporal_width
        )
        local_source_tokens = local_source_tokens.permute(0, 1, 3, 2, 4).contiguous()
        local_weights = step_mask[:, :, :, None, None].to(local_source_tokens.dtype)
        visit_source_tokens = (local_source_tokens * local_weights).sum(dim=2) / local_weights.sum(dim=2).clamp_min(1.0)
        local_event_tokens = local_source_tokens.mean(dim=3).contiguous()

        long_sequence = visit_source_tokens.permute(0, 2, 1, 3).contiguous()
        long_sequence = long_sequence + self.long_time_projection(inputs.long_time)[:, None]
        source_visit_mask = visit_mask[:, None, :].expand(-1, config.source_top_k, -1)
        long_sequence = long_sequence.reshape(batch * config.source_top_k, visits, config.temporal_width)
        long_sequence = self.long_temporal(
            long_sequence,
            mask=source_visit_mask.reshape(batch * config.source_top_k, visits),
        )
        source_long_time_tokens = long_sequence.reshape(
            batch, config.source_top_k, visits, config.temporal_width
        )
        long_weights = visit_mask[:, None, :, None].to(source_long_time_tokens.dtype)
        source_global_tokens = (source_long_time_tokens * long_weights).sum(dim=2) / long_weights.sum(dim=2).clamp_min(1.0)
        long_time_tokens = source_long_time_tokens.mean(dim=1).contiguous()
        source_pool_weights = self.source_pool_score(source_global_tokens).squeeze(-1).softmax(dim=1)
        pooled_long = (source_global_tokens * source_pool_weights.unsqueeze(-1)).sum(dim=1)

        period_features_by_source = self.period_proposal["features"](source_global_tokens)
        period_scores_by_source = self.period_proposal["scores"](source_global_tokens)
        constraint_logits_by_source = self.period_proposal["constraint"](source_global_tokens)
        period_features = self.period_proposal["features"](pooled_long)
        period_scores = self.period_proposal["scores"](pooled_long)
        period_posterior = period_scores.softmax(dim=-1)
        constraint_logits = self.period_proposal["constraint"](pooled_long)

        source_frame_event_logits = self.prediction_heads["event"](local_source_tokens).squeeze(-1)
        source_frame_event_logits = source_frame_event_logits * step_mask[:, :, :, None].to(
            source_frame_event_logits.dtype
        )
        source_frame_event_probability = source_frame_event_logits.sigmoid() * step_mask[:, :, :, None].to(
            source_frame_event_logits.dtype
        )
        source_event_logits = self.prediction_heads["source_event"](source_global_tokens).squeeze(-1)
        source_photometry_event_logits = self.source_photometry_event(
            source_photometry_values.reshape(frame_count, 2)
        ).reshape(batch, visits, steps, 1)
        photometry_weights = (wavelength_present & step_mask).unsqueeze(-1).to(
            source_photometry_event_logits.dtype
        )
        raw_photometry_count = photometry_weights.sum(dim=(1, 2))
        photometry_present = raw_photometry_count > 0
        photometry_count = raw_photometry_count.clamp_min(1.0)
        photometry_mean = (
            (source_photometry_event_logits * photometry_weights).sum(dim=(1, 2))
            / photometry_count
        )
        # A transit can occupy only a few cadences.  A plain temporal mean
        # dilutes that evidence and makes the direct photometry branch behave
        # like a constant-window classifier.  Use a normalized soft maximum
        # alongside the mean: when every frame agrees the result is unchanged,
        # while a short event remains visible without introducing another
        # learned parameter or changing checkpoint tensor shapes.
        temperature = 0.25
        masked_photometry_logits = source_photometry_event_logits.masked_fill(
            photometry_weights == 0, torch.finfo(source_photometry_event_logits.dtype).min
        )
        photometry_soft_max = temperature * (
            torch.logsumexp(masked_photometry_logits / temperature, dim=(1, 2))
            - photometry_count.log()
        )
        photometry_soft_max = torch.where(
            photometry_present,
            photometry_soft_max,
            torch.zeros_like(photometry_soft_max),
        )
        source_photometry_event_logits = 0.5 * (photometry_mean + photometry_soft_max)
        visit_source_event_logits = self.prediction_heads["visit_event"](visit_source_tokens).squeeze(-1)
        visit_source_event_logits = visit_source_event_logits * visit_mask[:, :, None].to(
            visit_source_event_logits.dtype
        )
        source_visit_event_logits = visit_source_event_logits.permute(0, 2, 1)
        visit_event_logits = visit_source_event_logits.max(dim=2).values
        pooled_backbone_event_logits = self.prediction_heads["global_event"](pooled_long).squeeze(-1)
        global_event_logits = combine_source_conditioned_event_logits(
            pooled_backbone_event_logits,
            source_event_logits,
            source_photometry_event_logits,
        )
        frame_event_logits = source_frame_event_logits.mean(dim=3)
        frame_event_probability = frame_event_logits.sigmoid()

        heatmaps = None
        if config.decode_heatmaps:
            source_heatmaps = self._decode_heatmaps(
                source_map,
                local_source_tokens.reshape(frame_count, config.source_top_k, config.temporal_width),
                raster,
                availability,
            )
            source_heatmaps = source_heatmaps.reshape(
                batch,
                visits,
                steps,
                config.source_top_k,
                config.canonical_wavelength_bins,
                config.heatmap_features,
                *config.heatmap_size,
            )
            source_heatmaps = source_heatmaps * step_mask[:, :, :, None, None, None, None, None].to(
                source_heatmaps.dtype
            )
            heatmap_weights = source_frame_event_probability / source_frame_event_probability.sum(
                dim=3, keepdim=True
            ).clamp_min(1e-6)
            heatmaps = (source_heatmaps * heatmap_weights[:, :, :, :, None, None, None, None]).sum(dim=3)
        else:
            source_heatmaps = None
            heatmaps = None
        source_logits_map = source_logits_map * step_mask[:, :, :, None, None].to(source_logits_map.dtype)
        source_heatmap = source_logits_map.sigmoid() * step_mask[:, :, :, None, None].to(source_logits_map.dtype)
        source_heatmap = source_heatmap.reshape(batch, visits, steps, *config.heatmap_size)
        source_event_heatmap = torch.zeros_like(source_heatmap).reshape(frame_count, -1)
        anchor_indices_flat = anchor_indices[:, None, None].expand(batch, visits, steps, -1).reshape(
            frame_count, config.source_top_k
        )
        source_event_heatmap.scatter_add_(1, anchor_indices_flat, source_frame_event_probability.reshape(frame_count, config.source_top_k))
        source_event_heatmap = source_event_heatmap.reshape(batch, visits, steps, *config.heatmap_size)
        candidate_heatmap = source_heatmap * source_event_heatmap

        visit_head_logits = {
            "candidate": self.prediction_heads["candidate"](visit_source_tokens.mean(dim=2)).squeeze(-1),
            "event": visit_event_logits,
            "artifact": self.prediction_heads["artifact"](visit_source_tokens.mean(dim=2)).squeeze(-1),
            "ood": self.prediction_heads["ood"](visit_source_tokens.mean(dim=2)).squeeze(-1),
            "coverage": self.prediction_heads["coverage"](visit_source_tokens.mean(dim=2)).squeeze(-1),
            "sufficiency": self.prediction_heads["sufficiency"](visit_source_tokens.mean(dim=2)).squeeze(-1),
        }
        visit_head_logits = {
            name: logits * visit_mask.to(logits.dtype)
            for name, logits in visit_head_logits.items()
        }
        global_head_logits = {
            "candidate": self.prediction_heads["candidate"](pooled_long).squeeze(-1),
            "event": global_event_logits,
            "artifact": self.prediction_heads["artifact"](pooled_long).squeeze(-1),
            "ood": self.prediction_heads["ood"](pooled_long).squeeze(-1),
            "coverage": self.prediction_heads["coverage"](pooled_long).squeeze(-1),
            "sufficiency": self.prediction_heads["sufficiency"](pooled_long).squeeze(-1),
        }
        head_logits = {
            **global_head_logits,
            "visit_event": visit_event_logits,
            "source_event": source_event_logits,
            "frame_event": frame_event_logits,
            "period_constraint": constraint_logits,
            "artifact": self.prediction_heads["artifact"](pooled_long).squeeze(-1),
            "ood": self.prediction_heads["ood"](pooled_long).squeeze(-1),
            "coverage": self.prediction_heads["coverage"](pooled_long).squeeze(-1),
            "sufficiency": self.prediction_heads["sufficiency"](pooled_long).squeeze(-1),
        }
        missing_modality_flags = torch.stack(
            (
                ~inputs.wavelength_mask.any(dim=(1, 2, 3)),
                ~inputs.object_mask.any(dim=(1, 2)),
            ),
            dim=-1,
        ).to(raster.dtype)

        global_head_names = {
            "candidate": "candidate_probability",
            "event": "event_probability",
            "artifact": "artifact_probability",
            "ood": "out_of_distribution_score",
            "coverage": "coverage_quality",
            "sufficiency": "data_sufficiency",
        }
        # Preserve the temporal coordinate for the requested transit-time
        # output.  The model input contract carries a normalized local-time
        # coordinate; conversion to BJD_TDB belongs to the dataset layer.
        local_step_time = inputs.local_time[..., 1]
        source_event_weights = source_frame_event_probability * step_mask[:, :, :, None].to(
            source_frame_event_probability.dtype
        )
        source_event_weight_sum = source_event_weights.sum(dim=(1, 2)).clamp_min(1e-6)
        transit_time_offset_by_source = (
            (source_event_weights * local_step_time[:, :, :, None]).sum(dim=(1, 2))
            / source_event_weight_sum
        )
        global_heads = {
            output_name: global_head_logits[name].sigmoid()
            for name, output_name in global_head_names.items()
        }
        global_heads["event_evidence_score"] = global_heads["event_probability"]
        return {
            "source_tokens": source_tokens_by_frame,
            "source_anchor_indices": anchor_indices,
            "source_anchor_xy": torch.stack(
                (
                    (anchor_indices % source_map.shape[-1]).to(raster.dtype) / max(source_map.shape[-1] - 1, 1),
                    (anchor_indices // source_map.shape[-1]).to(raster.dtype) / max(source_map.shape[-2] - 1, 1),
                ),
                dim=-1,
            ),
            "source_logits": source_logits_map,
            "background_tokens": background_tokens.reshape(
                batch, visits, steps, config.context_token_count, config.embedding_dim
            ),
            "object_context": object_context.reshape(
                batch, visits, steps, config.embedding_dim
            ),
            "geometry_coverage_condition": condition.reshape(
                batch, visits, steps, config.embedding_dim
            ),
            "wavelength_tokens": wavelength_embeddings.reshape(
                batch, visits, steps, wavelength_embeddings.shape[1], config.embedding_dim
            ),
            "wavelength_availability": availability.reshape(
                batch, visits, steps, config.canonical_wavelength_bins
            ),
            "local_event_tokens": local_event_tokens,
            "local_source_tokens": local_source_tokens,
            "long_time_tokens": long_time_tokens,
            "source_long_time_tokens": source_long_time_tokens,
            "source_event_logits": source_event_logits,
            "source_photometry_event_logits": source_photometry_event_logits,
            "visit_head_logits": visit_head_logits,
            "source_pool_weights": source_pool_weights,
            "visit_event_logits": visit_event_logits,
            "frame_event_logits": frame_event_logits,
            "global_event_logits": global_event_logits,
            "pooled_backbone_event_logits": pooled_backbone_event_logits,
            "head_logits": head_logits,
            "missing_modality_flags": missing_modality_flags,
            "period_proposal": {
                "scores": period_scores,
                "features": period_features,
                "grid_days": self.period_grid_days,
                "scores_by_source": period_scores_by_source,
                "features_by_source": period_features_by_source,
            },
            "heatmaps": heatmaps,
            "source_heatmaps": source_heatmaps,
            "heatmap_feature_names": HEATMAP_FEATURE_NAMES,
            "source_heatmap": source_heatmap,
            "candidate_heatmap": candidate_heatmap,
            "global_heads": global_heads,
            "orbit": {
                "period_posterior": period_posterior,
                "period_posterior_by_source": period_scores_by_source.softmax(dim=-1),
                "constraint_logits": constraint_logits,
                "constraint_logits_by_source": constraint_logits_by_source,
                # Both well- and weakly-constrained statuses expose a useful
                # period posterior.  The remaining statuses mean that the
                # period is not currently constrained.
                "period_is_constrained_probability": constraint_logits.softmax(dim=-1)[..., :2].sum(dim=-1),
                "period_is_constrained_probability_by_source": constraint_logits_by_source.softmax(dim=-1)[..., :2].sum(dim=-1),
                "transit_time_offset_by_source": transit_time_offset_by_source,
                "constraint_statuses": ORBIT_CONSTRAINT_STATUSES,
            },
            "diagnostics": {
                "source_scores": source_scores.reshape(batch, visits, steps, config.source_top_k),
                "source_frame_event_logits": source_frame_event_logits,
                "source_photometry": source_photometry.reshape(
                    batch, visits, steps, config.embedding_dim
                ),
                "measured_wavelength_mask": inputs.wavelength_mask,
                "object_mask": inputs.object_mask,
                "frame_event_probability": frame_event_probability,
                "source_visit_event_logits": source_visit_event_logits,
                "visit_mask": visit_mask,
                "step_mask": step_mask,
            },
        }

    def _decode_heatmaps(
        self,
        source_map: Tensor,
        frame_embedding: Tensor,
        raster: Tensor,
        availability: Tensor,
    ) -> Tensor:
        config = self.config
        spatial_features = self.spatial_temporal_decoder["feature_projection"](source_map)
        for block in self.spatial_temporal_decoder["refinement"]:
            spatial_features = block(spatial_features)
        spatial_factor = self.spatial_temporal_decoder["spatial_factor"](spatial_features)
        temporal_factor = self.spatial_temporal_decoder["temporal_factor"](frame_embedding)
        temporal_factor = temporal_factor.reshape(
            frame_embedding.shape[0],
            frame_embedding.shape[1],
            config.canonical_wavelength_bins,
            config.heatmap_features,
            config.heatmap_rank,
        )
        heatmaps = torch.einsum("nrhw,nkbfr->nkbfhw", spatial_factor, temporal_factor)

        validity = F.interpolate(raster[:, 3:4], size=config.heatmap_size, mode="area").clamp(0.0, 1.0)
        interpolation = F.interpolate(raster[:, 4:5], size=config.heatmap_size, mode="area").clamp(0.0, 1.0)
        source_presence = heatmaps[:, :, :, 2] + spatial_factor.mean(dim=1)[:, None, None]
        uncertainty = heatmaps[:, :, :, 3].abs()
        valid = (validity[:, None, None] * availability[:, None, :, None, None, None]).expand(
            -1, frame_embedding.shape[1], -1, -1, -1, -1
        )
        interpolated = (
            interpolation[:, None, None] * availability[:, None, :, None, None, None]
        ).expand(-1, frame_embedding.shape[1], -1, -1, -1, -1)
        return torch.cat(
            (
                heatmaps[:, :, :, :2],
                source_presence.unsqueeze(3),
                uncertainty.unsqueeze(3),
                valid,
                interpolated,
            ),
            dim=3,
        )
