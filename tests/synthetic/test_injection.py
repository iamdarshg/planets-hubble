import numpy as np

from synthetic.injection import RealParentInjector
from synthetic.parents import RealExposureParent, RealObservationParent


def make_parent() -> RealObservationParent:
    science = np.full((9, 9), 100.0, dtype=np.float32)
    science[4, 4] = 1000.0
    uncertainty = np.full((9, 9), 5.0, dtype=np.float32)
    dq = np.zeros((9, 9), dtype=np.uint16)
    exposure = RealExposureParent(
        exposure_id="exp-1",
        visit_id="visit-1",
        instrument="WFC3",
        detector="UVIS",
        filter_name="F606W",
        t_start_bjd_tdb=100.0,
        t_end_bjd_tdb=100.02,
        science=science,
        uncertainty=uncertainty,
        dq=dq,
    )
    return RealObservationParent(
        observation_id="obs-1",
        target_id="target-1",
        source_x=4.0,
        source_y=4.0,
        exposures=(exposure,),
        provenance={"source": "MAST"},
    )


def test_parent_injection_preserves_real_arrays_and_emits_transit_times() -> None:
    parent = make_parent()
    result = RealParentInjector().inject_transit(
        parent,
        epoch_bjd_tdb=100.01,
        period_days=4.0,
        depth=0.2,
        duration_days=0.1,
        source_flux_electrons=5000.0,
    )

    assert result.metadata["mode"] == "real_parent_injection"
    assert result.metadata["parent_observation_id"] == "obs-1"
    assert result.transit_times_bjd_tdb == (100.01,)
    np.testing.assert_array_equal(result.null[0].dq, result.injected[0].dq)
    np.testing.assert_array_equal(result.null[0].uncertainty, result.injected[0].uncertainty)
    assert np.any(result.injected[0].science < result.null[0].science)
    np.testing.assert_array_equal(parent.exposures[0].science, make_parent().exposures[0].science)


def test_parent_injection_does_not_invent_an_event_outside_window() -> None:
    result = RealParentInjector().inject_transit(
        make_parent(),
        epoch_bjd_tdb=101.0,
        period_days=4.0,
        depth=0.2,
        duration_days=0.01,
        source_flux_electrons=5000.0,
    )

    np.testing.assert_array_equal(result.null[0].science, result.injected[0].science)
    assert result.transit_times_bjd_tdb == ()
