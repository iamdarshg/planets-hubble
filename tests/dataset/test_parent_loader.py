import numpy as np

from dataset.models import ManifestRecord, WavelengthMetadata
from dataset.parent_loader import ManifestParentLoader


def test_manifest_loader_builds_parent_without_inventing_cadence_or_time() -> None:
    records = (
        ManifestRecord(
            observation_id="obs-1",
            product_id="prod-1",
            product_uri="mast:HST/product/a_flt.fits",
            download_uri="https://example/a.fits",
            observation_start=100.0,
            observation_midpoint=100.01,
            observation_end=100.02,
            time_system="BJD_TDB",
            exposure_duration_seconds=1728.0,
            wavelength=WavelengthMetadata(passband="F606W"),
            instrument="WFC3",
            spatial_footprint={"detector": "UVIS"},
        ),
    )

    def load_arrays(_record):
        return {
            "science": np.ones((4, 4), dtype=np.float32),
            "uncertainty": np.ones((4, 4), dtype=np.float32),
            "dq": np.zeros((4, 4), dtype=np.uint16),
            "pointing": {"roll_deg": 12.0},
        }

    parent = ManifestParentLoader(load_arrays).load(
        records, target_id="target-1", source_x=2.0, source_y=1.0
    )

    assert parent.exposures[0].t_start_bjd_tdb == 100.0
    assert parent.exposures[0].t_end_bjd_tdb == 100.02
    assert parent.exposures[0].exposure_seconds == 1728.0
    assert parent.exposures[0].provenance["product_id"] == "prod-1"
    assert parent.exposures[0].pointing["roll_deg"] == 12.0


def test_manifest_loader_requires_explicit_time_conversion_for_non_bjd() -> None:
    record = ManifestRecord(
        observation_id="obs-1",
        product_id=None,
        product_uri=None,
        download_uri=None,
        observation_start=59000.0,
        observation_end=59000.01,
        time_system="MJD",
        exposure_duration_seconds=864.0,
    )

    try:
        ManifestParentLoader(lambda _record: {"science": np.ones((2, 2))}).load(
            (record,), target_id="target", source_x=0.0, source_y=0.0
        )
    except ValueError as exc:
        assert "time_converter" in str(exc)
    else:
        raise AssertionError("non-BJD timestamps must not be silently relabeled")
