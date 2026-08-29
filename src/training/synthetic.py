"""Lazy synthetic batches for bounded AstroMamba-H pretraining smoke runs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import torch

from model import AstroMambaHInputs
from synthetic import SyntheticConfig, SyntheticGenerator

from .adapters import AstroMambaHTrainingBatch


def iter_synthetic_training_batches(
    config: SyntheticConfig,
    *,
    sample_count: int,
    device: torch.device | str = "cpu",
) -> Iterator[AstroMambaHTrainingBatch]:
    """Generate one model-ready synthetic sample at a time.

    The iterator intentionally retains no dataset-wide cache.  Each sample is
    generated, converted to the real AstroMamba-H input contract, yielded, and
    then becomes eligible for collection when the caller advances the stream.
    The model contract is fixed at 720x1280, so a full-resolution config is
    required here even though the standalone generator supports small arrays.
    Null and injected views alternate deterministically to provide both
    negative and positive pretraining examples without storing a manifest.
    """

    if not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    if (config.raster_height, config.raster_width) != (720, 1280):
        raise ValueError("synthetic training batches require a 720x1280 raster config")

    target_device = torch.device(device)
    for sample_index in range(sample_count):
        sample_config = replace(config, seed=config.seed + sample_index)
        bundle = SyntheticGenerator(sample_config).generate()
        view_name = "null" if sample_index % 2 else "injected"
        arrays = bundle.as_model_numpy(view_name)
        inputs = AstroMambaHInputs(
            **{
                name: torch.from_numpy(value).to(target_device)
                for name, value in arrays.items()
            }
        )
        view = bundle.null if view_name == "null" else bundle.injected
        target = torch.tensor(
            [[1.0 if view.labels.latent_positive else 0.0]],
            dtype=torch.float32,
            device=target_device,
        )
        yield AstroMambaHTrainingBatch(inputs=inputs, target=target)
