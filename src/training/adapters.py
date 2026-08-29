"""Small contract-shaped batches used by training smoke tests.

The real AstroMamba-H input contract contains 720x1280 rasters.  This adapter
keeps the modality names and masks while replacing the raster with a compact
summary, so local harness tests exercise optimization without allocating a
full science sample.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional

import torch
from torch import Tensor, nn

from model import AstroMambaH, AstroMambaHConfig, AstroMambaHInputs


@dataclass
class TinyAstroAdapterBatch:
    """A compact, normalized proxy for one AstroMamba-H batch.

    Dimensions mirror the feature widths in ``AstroMambaHConfig`` but are
    intentionally summary-sized.  It is not accepted by ``AstroMambaH``;
    production data should supply ``AstroMambaHInputs`` directly to the
    generic trainer.
    """

    raster_summary: Tensor
    wavelength_tokens: Tensor
    wavelength_mask: Tensor
    object_tokens: Tensor
    object_mask: Tensor
    geometry: Tensor
    exposure_duration: Tensor
    coverage_vector: Tensor
    local_time: Tensor
    long_time: Tensor
    target: Tensor

    @property
    def batch_size(self) -> int:
        return self.target.shape[0]

    def to(self, device: torch.device) -> "TinyAstroAdapterBatch":
        """Return a device-moved copy without mutating the source batch."""

        values = {
            field.name: getattr(self, field.name).to(device)
            for field in fields(self)
        }
        return replace(self, **values)


def make_tiny_adapter_batch(
    batch_size: int = 2,
    *,
    wavelength_count: int = 4,
    object_count: int = 3,
    local_steps: int = 2,
    device: Optional[torch.device | str] = None,
) -> TinyAstroAdapterBatch:
    """Create a deterministic-shape compact batch for smoke tests.

    The batch has no 720p allocation: ``raster_summary`` is ``[B, 6]`` and
    represents the six normalized raster channels after a future data-layer
    reduction.  Every variable-length modality retains an explicit boolean
    mask.
    """

    if batch_size < 1 or wavelength_count < 1 or object_count < 1 or local_steps < 1:
        raise ValueError("adapter batch dimensions must be positive")
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    wavelength_mask = torch.ones(batch_size, wavelength_count, dtype=torch.bool, device=target_device)
    object_mask = torch.ones(batch_size, object_count, dtype=torch.bool, device=target_device)
    if wavelength_count > 1:
        wavelength_mask[:, -1] = False
    if object_count > 1:
        object_mask[:, -1] = False

    raster_summary = torch.randn(batch_size, 6, device=target_device)
    wavelength_tokens = torch.randn(batch_size, wavelength_count, 8, device=target_device)
    wavelength_tokens[..., 0] = torch.linspace(
        0.0, 1.0, wavelength_count, device=target_device
    )
    object_tokens = torch.randn(batch_size, object_count, 12, device=target_device)
    geometry = torch.randn(batch_size, 10, device=target_device)
    exposure_duration = torch.rand(batch_size, 1, device=target_device) + 1.0
    coverage_vector = torch.rand(batch_size, 6, device=target_device)
    local_time = torch.randn(batch_size, local_steps, 5, device=target_device)
    long_time = torch.randn(batch_size, 5, device=target_device)
    target = (raster_summary[:, :1] > 0).to(torch.float32)
    return TinyAstroAdapterBatch(
        raster_summary=raster_summary,
        wavelength_tokens=wavelength_tokens,
        wavelength_mask=wavelength_mask,
        object_tokens=object_tokens,
        object_mask=object_mask,
        geometry=geometry,
        exposure_duration=exposure_duration,
        coverage_vector=coverage_vector,
        local_time=local_time,
        long_time=long_time,
        target=target,
    )


class TinyAstroAdapter(nn.Module):
    """A compact trainable adapter for local harness validation.

    This is deliberately not the AstroMamba-H research model.  It consumes
    the compact proxy above so CPU and GPU tests can validate optimization,
    AMP, and reporting without constructing a full raster model.
    """

    def __init__(self, hidden_width: int = 24) -> None:
        super().__init__()
        if hidden_width < 1:
            raise ValueError("hidden_width must be positive")
        self.projection = nn.Sequential(
            nn.Linear(53, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )

    @staticmethod
    def _masked_mean(tokens: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, batch: TinyAstroAdapterBatch) -> Tensor:
        wavelength = self._masked_mean(batch.wavelength_tokens, batch.wavelength_mask)
        objects = self._masked_mean(batch.object_tokens, batch.object_mask)
        local_time = batch.local_time.mean(dim=1)
        features = torch.cat(
            (
                batch.raster_summary,
                wavelength,
                objects,
                batch.geometry,
                batch.exposure_duration.log(),
                batch.coverage_vector,
                local_time,
                batch.long_time,
            ),
            dim=-1,
        )
        return self.projection(features)


@dataclass
class AstroMambaHTrainingBatch:
    """A labeled wrapper around the real AstroMamba-H input contract."""

    inputs: AstroMambaHInputs
    target: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.inputs.raster.shape[0])

    def to(self, device: torch.device) -> "AstroMambaHTrainingBatch":
        names = (
            "raster",
            "wavelength_tokens",
            "wavelength_mask",
            "object_tokens",
            "object_mask",
            "geometry",
            "exposure_duration",
            "coverage_vector",
            "local_time",
            "long_time",
            "coverage_map",
        )
        values = {
            name: getattr(self.inputs, name).to(device)
            if isinstance(getattr(self.inputs, name), Tensor)
            else getattr(self.inputs, name)
            for name in names
        }
        return AstroMambaHTrainingBatch(
            inputs=AstroMambaHInputs(**values),
            target=self.target.to(device),
        )


def tiny_astromamba_config() -> AstroMambaHConfig:
    """Return the smallest useful real-model configuration for GPU smoke."""

    return AstroMambaHConfig(
        stage_channels=(2, 3, 4, 5),
        embedding_dim=4,
        temporal_width=4,
        source_top_k=1,
        context_token_count=1,
        fusion_blocks=1,
        fusion_heads=1,
        temporal_blocks=1,
        canonical_wavelength_bins=2,
        wavelength_fourier_features=1,
        heatmap_rank=1,
        period_bin_count=2,
        period_feature_dim=2,
    )


def make_tiny_astromamba_batch(
    config: Optional[AstroMambaHConfig] = None,
    *,
    device: Optional[torch.device | str] = None,
) -> AstroMambaHTrainingBatch:
    """Create one full-raster batch for the CUDA smoke path."""

    config = config or tiny_astromamba_config()
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    batch, visits, steps, wavelengths, objects = 1, 1, 1, 2, 1
    raster = torch.zeros(
        batch, visits, steps, config.raster_channels, 720, 1280, device=target_device
    )
    wavelength_tokens = torch.zeros(
        batch, visits, steps, wavelengths, config.wavelength_feature_dim, device=target_device
    )
    wavelength_tokens[..., 0] = torch.tensor([0.2, 0.8], device=target_device)
    inputs = AstroMambaHInputs(
        raster=raster,
        wavelength_tokens=wavelength_tokens,
        wavelength_mask=torch.ones(
            batch, visits, steps, wavelengths, dtype=torch.bool, device=target_device
        ),
        object_tokens=torch.zeros(
            batch, visits, objects, config.object_feature_dim, device=target_device
        ),
        object_mask=torch.ones(
            batch, visits, objects, dtype=torch.bool, device=target_device
        ),
        geometry=torch.zeros(
            batch, visits, steps, config.geometry_feature_dim, device=target_device
        ),
        exposure_duration=torch.ones(batch, visits, steps, 1, device=target_device),
        coverage_vector=torch.ones(
            batch, visits, steps, config.coverage_feature_dim, device=target_device
        ),
        local_time=torch.zeros(
            batch, visits, steps, config.local_time_feature_dim, device=target_device
        ),
        long_time=torch.zeros(
            batch, visits, config.long_time_feature_dim, device=target_device
        ),
    )
    return AstroMambaHTrainingBatch(inputs=inputs, target=torch.ones(batch, 1, device=target_device))


class AstroMambaHTrainingAdapter(nn.Module):
    """Thin labeled-batch adapter that invokes the real AstroMamba-H model."""

    model_name = "AstroMambaH"

    def __init__(self, config: Optional[AstroMambaHConfig] = None) -> None:
        super().__init__()
        self.core = AstroMambaH(config or tiny_astromamba_config())

    def forward(self, batch: AstroMambaHTrainingBatch):
        return self.core(batch.inputs)
