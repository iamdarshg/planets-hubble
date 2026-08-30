import numpy as np
import pytest

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


def test_parent_injection_preserves_negative_calibrated_background() -> None:
    parent = make_parent()
    science = parent.exposures[0].science.copy()
    science[0, 0] = -20.0
    exposure = parent.exposures[0]
    parent = RealObservationParent(
        observation_id=parent.observation_id,
        target_id=parent.target_id,
        source_x=parent.source_x,
        source_y=parent.source_y,
        exposures=(
            RealExposureParent(
                exposure_id=exposure.exposure_id,
                visit_id=exposure.visit_id,
                instrument=exposure.instrument,
                detector=exposure.detector,
                filter_name=exposure.filter_name,
                t_start_bjd_tdb=exposure.t_start_bjd_tdb,
                t_end_bjd_tdb=exposure.t_end_bjd_tdb,
                science=science,
                uncertainty=exposure.uncertainty,
                dq=exposure.dq,
            ),
        ),
    )
    result = RealParentInjector().inject_transit(
        parent,
        epoch_bjd_tdb=100.01,
        period_days=4.0,
        depth=0.2,
        duration_days=0.1,
        source_flux_electrons=5000.0,
    )

    assert result.null[0].science[0, 0] == -20.0
    assert result.injected[0].science[0, 0] < 0.0
    assert np.all(result.injected[0].science <= result.null[0].science)


def test_parent_injection_integrates_over_explicit_exposure_duration_and_preserves_context() -> None:
    base = make_parent().exposures[0]
    wcs = {"crval": (180.0, 20.0), "ctype": ("RA---TAN", "DEC--TAN")}
    exposure = type(base)(
        exposure_id=base.exposure_id,
        visit_id=base.visit_id,
        instrument=base.instrument,
        detector=base.detector,
        filter_name=base.filter_name,
        t_start_bjd_tdb=base.t_start_bjd_tdb,
        t_end_bjd_tdb=base.t_end_bjd_tdb,
        exposure_duration_seconds=10.0,
        science=base.science,
        uncertainty=base.uncertainty,
        dq=base.dq,
        wcs=wcs,
        pointing={"roll_deg": 12.0},
        observer_position=(1.0, 2.0, 3.0),
        observer_velocity=(0.1, 0.2, 0.3),
        previous_exposure_fluence=np.ones((9, 9), dtype=np.float32),
        previous_exposure_time_bjd_tdb=99.0,
        provenance={"product": "flt", "source": "MAST"},
    )
    parent = type(make_parent())(
        observation_id="obs-context",
        target_id="target-context",
        source_x=4.0,
        source_y=4.0,
        exposures=(exposure,),
        object_tokens=np.array([[1.0, 2.0]], dtype=np.float32),
        object_mask=np.array([True]),
        object_metadata=({"object_id": "host", "role": "target_host"},),
        provenance={"source": "MAST", "archive_uri": "mast:obs-context"},
    )

    result = RealParentInjector().inject_transit(
        parent,
        epoch_bjd_tdb=exposure.t_mid_bjd_tdb - 2.0 / 86400.0,
        period_days=4.0,
        depth=0.2,
        duration_days=5.0 / 86400.0,
        source_flux_electrons=5000.0,
    )

    assert 0.0 < result.injected[0].relative_flux_drop < 0.2
    assert result.injected[0].metadata["wcs"] == wcs
    assert result.injected[0].metadata["pointing"] == {"roll_deg": 12.0}
    assert result.injected[0].metadata["observer_position"] == (1.0, 2.0, 3.0)
    assert result.injected[0].metadata["exposure_duration_seconds"] == 10.0
    assert result.metadata["realism_tier"] == "R4"
    assert result.metadata["r5_status"] == "external_only"
    assert result.metadata["source_context"]["source_xy"] == (4.0, 4.0)
    assert result.metadata["object_context"][0]["object_id"] == "host"
    np.testing.assert_array_equal(parent.exposures[0].science, make_parent().exposures[0].science)
    np.testing.assert_array_equal(parent.exposures[0].uncertainty, make_parent().exposures[0].uncertainty)
    np.testing.assert_array_equal(parent.exposures[0].dq, make_parent().exposures[0].dq)


def test_preserve_parent_requires_loaded_science_uncertainty_and_dq() -> None:
    parent = type(make_parent())(
        observation_id="obs-missing-arrays",
        target_id="target-missing-arrays",
        source_x=0.0,
        source_y=0.0,
        exposures=(
            type(make_parent().exposures[0])(
                exposure_id="empty",
                visit_id="visit",
                instrument="WFC3",
                detector="UVIS",
                filter_name="F606W",
                t_start_bjd_tdb=1.0,
                t_end_bjd_tdb=1.1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="science, uncertainty, and dq"):
        RealParentInjector().preserve_parent(parent)
