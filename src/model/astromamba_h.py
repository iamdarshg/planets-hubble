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

    def __post_init__(self) -> None:
        if (self.input_height, self.input_width) != (720, 1280):
            raise ValueError("AstroMamba-H requires a 720x1280 input raster")
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

    @property
    def input_size(self) -> Tuple[int, int]:
        return (self.input_height, self.input_width)

    @property
    def spatial_resolutions(self) -> Tuple[Tuple[int, int], ...]:
        """Feature resolutions in (height, width), matching SPEC.md."""

        return ((180, 320), (180, 320), (90, 160), (45, 80), (23, 40))

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

    def validate(self, config: AstroMambaHConfig) -> None:
        if self.raster.ndim != 6:
            raise ValueError("raster must have shape [B,V,S,C,720,1280]")
        b, visits, steps, channels, height, width = self.raster.shape
        if (height, width) != config.input_size:
            raise ValueError("raster must have spatial size 720x1280")
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

    def forward(self, tokens: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        coordinate = tokens[..., :1]
        argument = coordinate * self.frequencies.view(1, 1, -1) * torch.pi
        encoded = torch.cat((tokens, argument.sin(), argument.cos()), dim=-1)
        embeddings = self.projection(encoded)
        embeddings = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
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

    def forward(self, tokens: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        embeddings = self.projection(tokens)
        weights = mask.unsqueeze(-1).to(embeddings.dtype)
        embeddings = embeddings * weights
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
        wave_result, _ = self.wave_attention(
            query, safe_wave, safe_wave, key_padding_mask=self._safe_mask(wavelength_mask)
        )
        source = source + wave_result

        query = self.source_norm(source)
        object_values = self.object_norm(objects)
        safe_objects = object_values * object_mask.unsqueeze(-1).to(object_values.dtype)
        object_result, _ = self.object_attention(
            query, safe_objects, safe_objects, key_padding_mask=self._safe_mask(object_mask)
        )
        return source + object_result + self.feed_forward(source)


class GatedTemporalBlock(nn.Module):
    """Small core-PyTorch temporal mixer used in place of an unpinned Mamba package."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.in_projection = nn.Linear(width, width * 2)
        self.depthwise = nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width)
        self.out_projection = nn.Linear(width, width)

    def forward(self, sequence: Tensor) -> Tensor:
        normalized = self.norm(sequence)
        value, gate = self.in_projection(normalized).chunk(2, dim=-1)
        value = self.depthwise(value.transpose(1, 2)).transpose(1, 2)
        mixed = F.silu(value) * torch.sigmoid(gate)
        return sequence + self.out_projection(mixed)


class TemporalStage(nn.Module):
    def __init__(self, width: int, blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(GatedTemporalBlock(width) for _ in range(blocks))

    def forward(self, sequence: Tensor) -> Tensor:
        for block in self.blocks:
            sequence = block(sequence)
        return sequence


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
        self.cross_modal_fusion = nn.ModuleList(
            CrossModalBlock(config) for _ in range(config.fusion_blocks)
        )

        self.local_time_projection = nn.Linear(config.local_time_feature_dim, config.temporal_width)
        self.local_input_projection = nn.Linear(config.embedding_dim, config.temporal_width)
        self.local_temporal = TemporalStage(config.temporal_width, config.temporal_blocks)
        self.long_time_projection = nn.Linear(config.long_time_feature_dim, config.temporal_width)
        self.long_temporal = TemporalStage(config.temporal_width, config.temporal_blocks)

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

        self.spatial_temporal_decoder = nn.ModuleDict(
            {
                "spatial_factor": nn.Conv2d(c1, config.heatmap_rank, kernel_size=1),
                "temporal_factor": nn.Linear(
                    config.temporal_width,
                    config.canonical_wavelength_bins
                    * config.heatmap_features
                    * config.heatmap_rank,
                ),
            }
        )
        self.prediction_heads = nn.ModuleDict(
            {
                "candidate": nn.Linear(config.temporal_width, 1),
                "event": nn.Linear(config.temporal_width, 1),
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
            "spatial_backbone": self.spatial_backbone,
            "source_tokenizer": self.source_tokenizer,
            "wavelength_encoder": self.wavelength_encoder,
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

    def forward(self, inputs: AstroMambaHInputs) -> Dict[str, object]:
        inputs.validate(self.config)
        config = self.config
        batch, visits, steps, _, _, _ = inputs.raster.shape
        frame_count = batch * visits * steps

        raster = inputs.raster.reshape(frame_count, config.raster_channels, *config.input_size)
        _, _, source_map, _, _ = self.spatial_backbone(raster)
        source_tokens, background_tokens, source_scores, source_logits_map = self.source_tokenizer(
            source_map
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

        source_tokens = source_tokens + condition.unsqueeze(1) + wavelength_context.unsqueeze(1)
        source_tokens = source_tokens + object_context.unsqueeze(1)
        for block in self.cross_modal_fusion:
            source_tokens = block(
                source_tokens,
                wavelength_embeddings,
                wavelength_mask,
                object_embeddings,
                object_mask,
            )

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
        local_sequence = self.local_temporal(local_sequence + local_time_embedding)
        local_source_tokens = local_sequence.reshape(
            batch, visits, config.source_top_k, steps, config.temporal_width
        )
        local_event_tokens = local_source_tokens.mean(dim=2).contiguous()

        long_sequence = local_event_tokens.mean(dim=2)
        long_sequence = long_sequence + self.long_time_projection(inputs.long_time)
        long_time_tokens = self.long_temporal(long_sequence)
        pooled_long = long_time_tokens.mean(dim=1)

        period_features = self.period_proposal["features"](pooled_long)
        period_scores = self.period_proposal["scores"](pooled_long)
        period_posterior = period_scores.softmax(dim=-1)
        constraint_logits = self.period_proposal["constraint"](pooled_long)

        frame_event_logits = self.prediction_heads["frame_event"](local_event_tokens).squeeze(-1)
        frame_event_probability = frame_event_logits.sigmoid()

        heatmaps = self._decode_heatmaps(
            source_map,
            local_event_tokens.reshape(frame_count, config.temporal_width),
            raster,
            availability,
        )
        heatmaps = heatmaps.reshape(
            batch,
            visits,
            steps,
            config.canonical_wavelength_bins,
            config.heatmap_features,
            *config.heatmap_size,
        )
        source_heatmap = source_logits_map.sigmoid().reshape(batch, visits, steps, *config.heatmap_size)
        candidate_heatmap = source_heatmap * frame_event_probability[..., None, None]

        global_head_names = {
            "candidate": "candidate_probability",
            "event": "event_probability",
            "artifact": "artifact_probability",
            "ood": "out_of_distribution_score",
            "coverage": "coverage_quality",
            "sufficiency": "data_sufficiency",
        }
        global_heads = {
            output_name: self.prediction_heads[name](long_time_tokens).squeeze(-1).sigmoid()
            for name, output_name in global_head_names.items()
        }
        global_heads["event_evidence_score"] = global_heads["event_probability"]
        return {
            "source_tokens": source_tokens_by_frame,
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
            "long_time_tokens": long_time_tokens,
            "period_proposal": {
                "scores": period_scores,
                "features": period_features,
                "grid_days": self.period_grid_days,
            },
            "heatmaps": heatmaps,
            "heatmap_feature_names": HEATMAP_FEATURE_NAMES,
            "source_heatmap": source_heatmap,
            "candidate_heatmap": candidate_heatmap,
            "global_heads": global_heads,
            "orbit": {
                "period_posterior": period_posterior,
                "constraint_logits": constraint_logits,
                "constraint_statuses": ORBIT_CONSTRAINT_STATUSES,
            },
            "diagnostics": {
                "source_scores": source_scores.reshape(batch, visits, steps, config.source_top_k),
                "measured_wavelength_mask": inputs.wavelength_mask,
                "object_mask": inputs.object_mask,
                "frame_event_probability": frame_event_probability,
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
        spatial_factor = self.spatial_temporal_decoder["spatial_factor"](source_map)
        temporal_factor = self.spatial_temporal_decoder["temporal_factor"](frame_embedding)
        temporal_factor = temporal_factor.reshape(
            frame_embedding.shape[0],
            config.canonical_wavelength_bins,
            config.heatmap_features,
            config.heatmap_rank,
        )
        heatmaps = torch.einsum("nrhw,nbfr->nbfhw", spatial_factor, temporal_factor)

        validity = F.interpolate(raster[:, 3:4], size=config.heatmap_size, mode="area").clamp(0.0, 1.0)
        interpolation = F.interpolate(raster[:, 4:5], size=config.heatmap_size, mode="area").clamp(0.0, 1.0)
        heatmaps[:, :, 2] = heatmaps[:, :, 2] + spatial_factor.mean(dim=1).unsqueeze(1)
        heatmaps[:, :, 3] = heatmaps[:, :, 3].abs()
        heatmaps[:, :, 4] = validity * availability[:, :, None, None]
        heatmaps[:, :, 5] = interpolation * availability[:, :, None, None]
        return heatmaps
