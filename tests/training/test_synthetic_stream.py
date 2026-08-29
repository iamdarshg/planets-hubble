from __future__ import annotations

import torch

from synthetic import SyntheticConfig
from training.synthetic import iter_synthetic_training_batches


def test_synthetic_training_stream_is_lazy_and_emits_model_batches() -> None:
    config = SyntheticConfig(
        seed=101,
        visits=1,
        local_steps=1,
        raster_height=720,
        raster_width=1280,
        wavelength_nm=(450.0, 650.0),
        timestamp_jitter_days=0.0,
    )

    stream = iter_synthetic_training_batches(config, sample_count=2, device="cpu")
    first = next(stream)

    assert first.batch_size == 1
    assert first.inputs.raster.shape == (1, 1, 1, 6, 720, 1280)
    assert first.inputs.wavelength_tokens.shape == (1, 1, 1, 2, 8)
    assert first.inputs.wavelength_mask.dtype == torch.bool
    assert torch.isfinite(first.target).all()

    second = next(stream)
    assert second.inputs.raster.shape == first.inputs.raster.shape
    assert not torch.equal(first.inputs.wavelength_tokens, second.inputs.wavelength_tokens)

    try:
        next(stream)
    except StopIteration:
        pass
    else:
        raise AssertionError("bounded synthetic stream yielded more than sample_count batches")
