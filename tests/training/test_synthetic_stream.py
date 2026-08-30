from __future__ import annotations

import numpy as np
import torch

from synthetic import RealExposureParent, RealObservationParent, SyntheticConfig
from training import synthetic as training_synthetic
from training.synthetic import (
    iter_parented_synthetic_training_batches,
    iter_paired_synthetic_training_batches,
    iter_synthetic_training_batches,
)


def test_disk_cache_persists_lru_entries_without_retaining_arrays(tmp_path) -> None:
    ProceduralSyntheticCache = training_synthetic.ProceduralSyntheticCache
    cache_dir = tmp_path / "synthetic-cache"
    cache = ProceduralSyntheticCache(
        cache_dir,
        max_cache_entries=2,
        max_cache_bytes=8_192,
        max_entry_bytes=4_096,
    )

    assert cache.store("first", {"values": np.arange(64, dtype=np.float32)}, {"seed": 1})
    assert cache.store("second", {"values": np.arange(32, dtype=np.float32)}, {"seed": 2})
    first = cache.load("first")
    assert first is not None
    assert first.metadata == {"seed": 1}
    assert np.array_equal(first.arrays["values"], np.arange(64, dtype=np.float32))

    assert cache.store("third", {"values": np.arange(16, dtype=np.float32)}, {"seed": 3})

    assert cache.max_cache_entries == 2
    assert cache.cache_size == 2
    assert cache.cache_bytes <= cache.max_cache_bytes
    assert len(tuple(cache_dir.glob("*.npz"))) == 2
    assert cache.load("second") is None
    assert all(not isinstance(entry, np.ndarray) for entry in cache._entries.values())

    restarted = ProceduralSyntheticCache(cache_dir, max_cache_entries=2, max_cache_bytes=8_192)
    restored = restarted.load("first")
    assert restored is not None
    assert restored.metadata == {"seed": 1}
    assert np.array_equal(restored.arrays["values"], np.arange(64, dtype=np.float32))


def test_disk_cache_rejects_entries_over_its_byte_limit(tmp_path) -> None:
    cache = training_synthetic.ProceduralSyntheticCache(
        tmp_path / "synthetic-cache",
        max_cache_entries=64,
        max_cache_bytes=1_024,
        max_entry_bytes=128,
    )

    assert not cache.store("too-large", {"values": np.arange(1_024, dtype=np.float32)}, {})
    assert cache.cache_size == 0
    assert not tuple((tmp_path / "synthetic-cache").glob("*.npz"))


def test_synthetic_training_stream_is_lazy_and_emits_model_batches(tmp_path, monkeypatch) -> None:
    config = SyntheticConfig(
        seed=101,
        visits=1,
        local_steps=1,
        raster_height=720,
        raster_width=1280,
        wavelength_nm=(450.0, 650.0),
        timestamp_jitter_days=0.0,
    )

    cache_dir = tmp_path / "synthetic-cache"
    stream = iter_synthetic_training_batches(
        config,
        sample_count=2,
        device="cpu",
        cache_dir=cache_dir,
        max_cache_entries=64,
    )
    assert training_synthetic.ProceduralSyntheticCache(cache_dir).cache_size == 0
    first = next(stream)

    assert training_synthetic.ProceduralSyntheticCache(cache_dir).cache_size == 1
    assert first.batch_size == 1
    assert first.inputs.raster.shape == (1, 1, 1, 6, 720, 1280)
    assert first.inputs.wavelength_tokens.shape == (1, 1, 1, 2, 8)
    assert first.inputs.wavelength_mask.dtype == torch.bool
    assert torch.isfinite(first.target).all()
    assert first.auxiliary_targets["candidate_heatmap"].shape == (1, 1, 1, 90, 160)
    assert first.auxiliary_targets["source"].shape == (1, 1, 1, 90, 160)
    assert first.auxiliary_targets["candidate_heatmap"].sum() > 0
    assert torch.equal(
        first.auxiliary_targets["source"], first.auxiliary_targets["candidate_heatmap"]
    )
    assert "artifact" not in first.auxiliary_targets
    assert "ood" not in first.auxiliary_targets
    assert "period_constraint" in first.auxiliary_targets
    assert first.auxiliary_targets["period_constraint"].tolist() == [1]

    second = next(stream)
    assert training_synthetic.ProceduralSyntheticCache(cache_dir).cache_size == 2
    assert second.inputs.raster.shape == first.inputs.raster.shape
    assert not torch.equal(first.inputs.wavelength_tokens, second.inputs.wavelength_tokens)
    assert torch.count_nonzero(second.auxiliary_targets["candidate_heatmap"]) == 0
    assert torch.count_nonzero(second.auxiliary_targets["source"]) > 0
    assert "period_constraint" not in second.auxiliary_targets

    try:
        next(stream)
    except StopIteration:
        pass
    else:
        raise AssertionError("bounded synthetic stream yielded more than sample_count batches")

    del first, second

    def generation_must_not_run(_self):
        raise AssertionError("cached synthetic stream generated instead of loading its SSD entry")

    monkeypatch.setattr(training_synthetic.SyntheticGenerator, "generate", generation_must_not_run)
    restarted = next(
        iter_synthetic_training_batches(
            config,
            sample_count=1,
            device="cpu",
            cache_dir=cache_dir,
            max_cache_entries=64,
        )
    )
    assert restarted.inputs.raster.shape == (1, 1, 1, 6, 720, 1280)


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


def test_paired_stream_keeps_null_and_injected_counterfactuals_together(tmp_path, monkeypatch) -> None:
    config = SyntheticConfig(
        seed=9,
        visits=1,
        local_steps=1,
        raster_height=720,
        raster_width=1280,
        timestamp_jitter_days=0.0,
    )
    cache_dir = tmp_path / "synthetic-cache"
    batch = next(
        iter_paired_synthetic_training_batches(
            config,
            sample_count=1,
            device="cpu",
            cache_dir=cache_dir,
            max_cache_entries=64,
        )
    )

    assert training_synthetic.ProceduralSyntheticCache(cache_dir).cache_size == 1
    assert batch.inputs.raster.shape == (2, 1, 1, 6, 720, 1280)
    assert batch.target[:, 0].tolist() == [0.0, 1.0]
    assert batch.auxiliary_targets["frame_event"].shape == (2, 1, 1)
    for name in (
        "geometry",
        "exposure_duration",
        "coverage_vector",
        "local_time",
        "long_time",
        "object_tokens",
        "object_mask",
        "wavelength_mask",
    ):
        values = getattr(batch.inputs, name)
        assert torch.equal(values[0], values[1]), name
    assert torch.equal(
        batch.inputs.raster[0, :, :, 2:, :, :],
        batch.inputs.raster[1, :, :, 2:, :, :],
    )
    assert torch.equal(
        batch.inputs.wavelength_tokens[0, ..., (0, 3, 4, 5, 6, 7)],
        batch.inputs.wavelength_tokens[1, ..., (0, 3, 4, 5, 6, 7)],
    )
    assert torch.equal(batch.auxiliary_targets["source"][0], batch.auxiliary_targets["source"][1])
    assert torch.equal(
        batch.auxiliary_targets["candidate_heatmap"][0],
        batch.auxiliary_targets["source"][0] * batch.auxiliary_targets["frame_event"][0, ..., None, None],
    )

    del batch

    def generation_must_not_run(_self):
        raise AssertionError("paired cache did not preserve the generated nuisance realization")

    monkeypatch.setattr(training_synthetic.SyntheticGenerator, "generate", generation_must_not_run)
    restored = next(
        iter_paired_synthetic_training_batches(
            config,
            sample_count=1,
            device="cpu",
            cache_dir=cache_dir,
            max_cache_entries=64,
        )
    )
    assert torch.equal(
        restored.inputs.raster[0, :, :, 2:, :, :],
        restored.inputs.raster[1, :, :, 2:, :, :],
    )


def test_synthetic_quality_targets_follow_observability_masks() -> None:
    config = SyntheticConfig(
        seed=44,
        visits=1,
        local_steps=2,
        raster_height=720,
        raster_width=1280,
        invalid_exposures=((0, 1),),
        timestamp_jitter_days=0.0,
    )
    batch = next(iter_synthetic_training_batches(config, sample_count=1, device="cpu"))

    assert batch.auxiliary_targets["coverage"].item() == 0.5
    assert batch.auxiliary_targets["sufficiency"].item() == 0.5
