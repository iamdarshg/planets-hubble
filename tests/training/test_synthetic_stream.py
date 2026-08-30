from __future__ import annotations

import torch
import numpy as np

from synthetic import RealExposureParent, RealObservationParent, SyntheticConfig
from training.synthetic import (
    iter_parented_synthetic_training_batches,
    iter_paired_synthetic_training_batches,
    iter_synthetic_training_batches,
)


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
                exposure_duration_seconds=12.5,
                science=image,
                uncertainty=np.ones_like(image),
                dq=np.zeros_like(image, dtype=np.uint16),
            ),
        ),
    )
    batch = next(iter_parented_synthetic_training_batches((parent,), sample_count=1, device="cpu"))
    assert batch.inputs.raster.shape == (1, 1, 1, 6, 720, 1280)
    assert batch.inputs.wavelength_tokens.shape[-1] == 8
    assert batch.inputs.exposure_duration.item() == 12.5
    assert batch.inputs.wavelength_tokens[0, 0, 0, 0, 4].item() == 12.5
    assert batch.target.item() == 1.0
    assert batch.auxiliary_targets["frame_event"].any()
    assert batch.auxiliary_targets["source_event"].shape == (1, 96)
    assert batch.auxiliary_targets["source_event"][0, 0].item() == 1.0

    summary = next(
        iter_parented_synthetic_training_batches(
            (parent,), sample_count=1, device="cpu", sequence_summary=True
        )
    )
    assert summary.inputs.raster.shape == (1, 1, 1, 6, 720, 1280)
    assert summary.inputs.local_time.shape == (1, 1, 1, 5)
    assert summary.inputs.long_time.shape == (1, 1, 5)
    assert summary.inputs.step_mask.shape == (1, 1, 1)
    assert summary.target.item() == 1.0


def test_parented_stream_labels_null_counterfactual_as_negative() -> None:
    image = np.full((720, 1280), 100.0, dtype=np.float32)
    parent = RealObservationParent(
        observation_id="label-parent",
        target_id="label-target",
        source_x=640.0,
        source_y=360.0,
        exposures=(
            RealExposureParent(
                exposure_id="label-exp",
                visit_id="label-visit",
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
    batches = iter_parented_synthetic_training_batches((parent,), sample_count=2, device="cpu")
    next(batches)
    null_batch = next(batches)
    assert null_batch.target.item() == 0.0
    assert not null_batch.auxiliary_targets["frame_event"].any()
    assert null_batch.auxiliary_targets["source_event"][0, 0].item() == 0.0


def test_paired_stream_keeps_null_and_injected_counterfactuals_together() -> None:
    config = SyntheticConfig(
        seed=9,
        visits=1,
        local_steps=1,
        raster_height=720,
        raster_width=1280,
        timestamp_jitter_days=0.0,
    )
    batch = next(iter_paired_synthetic_training_batches(config, sample_count=1, device="cpu"))

    assert batch.inputs.raster.shape == (2, 1, 1, 6, 720, 1280)
    assert batch.target[:, 0].tolist() == [0.0, 1.0]
    assert batch.auxiliary_targets["frame_event"].shape == (2, 1, 1)
