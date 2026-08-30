from __future__ import annotations

import numpy as np
import pytest
from dataclasses import replace

from synthetic.parents import RealExposureParent, RealObservationParent
from synthetic.schedules import ObservationScheduleSampler


def make_parent() -> RealObservationParent:
    exposures = (
        RealExposureParent(
            exposure_id="exp-1",
            visit_id="visit-1",
            instrument="WFC3",
            detector="UVIS",
            filter_name="F606W",
            t_start_bjd_tdb=2460000.1,
            t_end_bjd_tdb=2460000.11,
            science=np.zeros((8, 8), dtype=np.float32),
            uncertainty=np.ones((8, 8), dtype=np.float32),
            dq=np.zeros((8, 8), dtype=np.uint16),
            provenance={"source": "MAST", "observation_id": "obs-1"},
        ),
        RealExposureParent(
            exposure_id="exp-2",
            visit_id="visit-2",
            instrument="WFC3",
            detector="IR",
            filter_name="F160W",
            t_start_bjd_tdb=2460002.0,
            t_end_bjd_tdb=2460002.03,
            science=np.zeros((8, 8), dtype=np.float32),
            uncertainty=np.ones((8, 8), dtype=np.float32),
            dq=np.zeros((8, 8), dtype=np.uint16),
            provenance={"source": "MAST", "observation_id": "obs-2"},
        ),
    )
    return RealObservationParent(
        observation_id="obs-1",
        target_id="target-1",
        source_x=3.5,
        source_y=3.5,
        exposures=exposures,
        provenance={"source": "MAST", "archive_uri": "mast:HST/obs-1"},
    )


def test_parent_preserves_exposure_windows_and_provenance() -> None:
    parent = make_parent()

    assert parent.exposures[0].t_start_bjd_tdb == 2460000.1
    assert parent.exposures[0].t_end_bjd_tdb == 2460000.11
    assert parent.exposures[0].provenance["source"] == "MAST"
    assert parent.exposures[1].detector == "IR"


def test_schedule_replay_preserves_gaps_without_timestamp_jitter() -> None:
    schedule = ObservationScheduleSampler(make_parent()).sample()

    np.testing.assert_allclose(schedule.starts, [2460000.1, 2460002.0])
    np.testing.assert_allclose(schedule.ends, [2460000.11, 2460002.03])
    np.testing.assert_allclose(
        schedule.mids,
        [(2460000.1 + 2460000.11) / 2.0, (2460002.0 + 2460002.03) / 2.0],
    )
    assert schedule.metadata["mode"] == "real_parent_replay"
    assert schedule.metadata["gap_days"] > 1.0


def test_schedule_replay_preserves_physical_exposure_duration_and_read_only_arrays() -> None:
    first = replace(make_parent().exposures[0], exposure_duration_seconds=12.5)
    parent = replace(make_parent(), exposures=(first, make_parent().exposures[1]))

    schedule = ObservationScheduleSampler(parent).sample()

    np.testing.assert_array_equal(schedule.exposure_duration_seconds, [12.5, parent.exposures[1].exposure_seconds])
    assert not schedule.exposure_duration_seconds.flags.writeable
    with pytest.raises(ValueError):
        schedule.exposure_duration_seconds[0] = 1.0


def test_block_bootstrap_selects_whole_visits_and_is_reproducible() -> None:
    parent = make_parent()
    first = ObservationScheduleSampler(parent).block_bootstrap(seed=12, visits=3)
    second = ObservationScheduleSampler(parent).block_bootstrap(seed=12, visits=3)

    np.testing.assert_array_equal(first.exposure_ids, second.exposure_ids)
    assert len(first.exposure_ids) == 3
    assert set(first.exposure_ids).issubset({"exp-1", "exp-2"})
    assert first.metadata["mode"] == "whole_visit_block_bootstrap"


def test_parent_rejects_non_monotonic_or_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="t_end"):
        RealExposureParent(
            exposure_id="bad",
            visit_id="visit",
            instrument="WFC3",
            detector="UVIS",
            filter_name="F606W",
            t_start_bjd_tdb=2.0,
            t_end_bjd_tdb=1.0,
        )

    with pytest.raises(ValueError, match="science"):
        RealExposureParent(
            exposure_id="bad",
            visit_id="visit",
            instrument="WFC3",
            detector="UVIS",
            filter_name="F606W",
            t_start_bjd_tdb=1.0,
            t_end_bjd_tdb=2.0,
            science=np.zeros((4, 4)),
            uncertainty=np.zeros((3, 3)),
        )
