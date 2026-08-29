from __future__ import annotations

import torch
import numpy as np

from synthetic import RealExposureParent, RealObservationParent, SyntheticConfig
from training.synthetic import iter_parented_synthetic_training_batches, iter_synthetic_training_batches


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


def test_parented_stream_emits_full_resolution_model_batches() -> None:
    image = np.full((720, 1280), 100.0, dtype=np.float32)
    parent = RealObservationParent(
        observation_id="stream-parent",
        target_id="stream-target",
        source_x=640.0,
        source_y=360.0,
        exposures=(
            RealExposureParent(
                exposure_id="stream-exp",
                visit_id="stream-visit",
                instrument="WFC3",
                detector="UVIS",
                filter_name="F606W",
                t_start_bjd_tdb=20.0,
                t_end_bjd_tdb=20.01,
                science=image,
                uncertainty=np.ones_like(image),
                dq=np.zeros_like(image, dtype=np.uint16),
            ),
        ),
    )
    batch = next(iter_parented_synthetic_training_batches((parent,), sample_count=1, device="cpu"))
    assert batch.inputs.raster.shape == (1, 1, 1, 6, 720, 1280)
    assert batch.inputs.wavelength_tokens.shape[-1] == 8
