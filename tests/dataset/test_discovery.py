from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import parse_qs
import json
from typing import Any

import pytest

from dataset.mast import DiscoveryFilters, MastDiscoveryClient, MastResponseError
from dataset.transport import MastJsonTransport


class FixtureTransport:
    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((url, payload))
        if not self.responses:
            raise AssertionError("unexpected extra MAST request")
        return self.responses.pop(0)


def _observation(
    *,
    obsid: str,
    rights: str = "public",
    instrument: str = "WFC3/UVIS",
    product_type: str = "IMAGE",
) -> dict[str, Any]:
    return {
        "obsid": obsid,
        "obs_id": f"{obsid}-RAW",
        "obs_collection": "HST",
        "instrument_name": instrument,
        "dataproduct_type": product_type,
        "calib_level": 3,
        "dataRights": rights,
        "target_name": "Vega",
        "t_min": 60000.0,
        "t_max": 60000.25,
        "t_exptime": 900.0,
        "s_ra": 279.2347,
        "s_dec": 38.7837,
        "s_region": "Circle ICRS 279.2347 38.7837 0.01",
        "em_min": 200.0,
        "em_max": 1000.0,
        "filters": "F275W",
        "proposal_id": "12345",
        "dataURL": "mast:HST/observation-product",
        "quality_flag": "OK",
        "exposure_count": 1,
    }


def _product(obsid: str, *, rights: str = "public") -> dict[str, Any]:
    return {
        "obsid": obsid,
        "obs_id": f"{obsid}-DRZ",
        "productFilename": f"{obsid}_drz.fits",
        "productType": "SCIENCE",
        "productSubGroupDescription": "DRZ",
        "calib_level": 3,
        "dataRights": rights,
        "dataURI": f"mast:HST/product/{obsid}_drz.fits",
        "dataURL": "https://mast.stsci.edu/download/file",
        "em_min": 250.0,
        "em_max": 900.0,
        "filters": "F275W",
        "t_exptime": 900.0,
        "wcs_uri": "mast:HST/product/wcs.json",
        "observer_position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "observer_velocity": {"x": 4.0, "y": 5.0, "z": 6.0},
        "pointing": {"roll_deg": 12.5, "boresight_ra_deg": 279.2347},
        "coverage": {"valid_fraction": 0.98},
    }


def test_named_target_resolves_position_searches_and_expands_public_products() -> None:
    transport = FixtureTransport(
        [
            {
                "resolvedCoordinate": [
                    {"canonicalName": "VEGA", "ra": 279.2347, "decl": 38.7837}
                ]
            },
            {"data": [_observation(obsid="obs-1")]},
            {"data": [_product("obs-1"), _product("obs-1-private", rights="restricted")]},
        ]
    )

    manifest = MastDiscoveryClient(transport).discover_named_target(
        "Vega", patch_id="vega-patch", radius_deg=0.05
    )

    assert len(transport.calls) == 3
    assert transport.calls[0][1]["service"] == "Mast.Name.Lookup"
    assert transport.calls[1][1]["service"] == "Mast.Caom.Cone"
    assert transport.calls[1][1]["params"] == {
        "ra": 279.2347,
        "dec": 38.7837,
        "radius": 0.05,
    }
    assert transport.calls[2][1] == {
        "service": "Mast.Caom.Products",
        "params": {"obsid": "obs-1"},
        "format": "json",
        "pagesize": 2000,
        "page": 1,
        "removenullcolumns": True,
    }

    record = manifest.records[0]
    assert manifest.patch_id == "vega-patch"
    assert manifest.source_identifier == "VEGA"
    assert manifest.target_position.ra_deg == pytest.approx(279.2347)
    assert record.observation_id == "obs-1"
    assert record.product_uri == "mast:HST/product/obs-1_drz.fits"
    assert record.download_uri == "https://mast.stsci.edu/download/file"
    assert record.start_time == 60000.0
    assert record.midpoint_time == 60000.125
    assert record.end_time == 60000.25
    assert record.time_system == "MJD"
    assert record.exposure_duration_seconds == 900.0
    assert record.wavelength.minimum_nm == 250.0
    assert record.wavelength.maximum_nm == 900.0
    assert record.wavelength.passband == "F275W"
    assert record.wcs_uri == "mast:HST/product/wcs.json"
    assert record.observer_position == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert record.pointing["roll_deg"] == 12.5
    assert record.coverage["valid_fraction"] == 0.98
    assert record.spatial_footprint["s_region"].startswith("Circle ICRS")
    assert len(manifest.to_dict()["records"]) == 1


def test_sky_patch_can_use_filtered_position_and_filters_public_science_rows() -> None:
    transport = FixtureTransport(
        [
            {
                "data": [
                    _observation(obsid="public", instrument="WFC3/UVIS"),
                    _observation(obsid="private", rights="exclusive_access"),
                    _observation(obsid="wrong-instrument", instrument="ACS/WFC"),
                    _observation(obsid="wrong-type", product_type="SPECTRUM"),
                ]
            },
            {"data": [_product("public")]},
        ]
    )

    manifest = MastDiscoveryClient(transport).discover_sky_patch(
        patch_id="patch-1",
        ra_deg=10.0,
        dec_deg=-20.0,
        radius_deg=0.2,
        instruments=("WFC3/UVIS",),
        product_types=("IMAGE",),
        search_service="Mast.Caom.Filtered.Position",
    )

    request = transport.calls[0][1]
    assert request["service"] == "Mast.Caom.Filtered.Position"
    assert request["params"]["position"] == "10.0, -20.0, 0.2"
    assert {item["paramName"] for item in request["params"]["filters"]} == {
        "dataRights",
        "instrument_name",
        "dataproduct_type",
    }
    assert [record.observation_id for record in manifest.records] == ["public"]


def test_timeseries_helpers_add_an_explicit_caom_timeseries_filter() -> None:
    transport = FixtureTransport(
        [
            {"data": [_observation(obsid="series", product_type="TIMESERIES")]},
            {"data": [_product("series")]},
        ]
    )

    manifest = MastDiscoveryClient(transport).discover_time_series_sky_patch(
        patch_id="series-patch",
        ra_deg=10.0,
        dec_deg=-20.0,
        radius_deg=0.2,
        instruments=("WFC3/UVIS",),
    )

    request = transport.calls[0][1]
    assert request["service"] == "Mast.Caom.Filtered.Position"
    assert {item["paramName"]: item["values"] for item in request["params"]["filters"]}["dataproduct_type"] == ["TIMESERIES"]
    assert manifest.records[0].product_type == "TIMESERIES"


def test_timeseries_helper_rejects_a_non_timeseries_product_type() -> None:
    with pytest.raises(ValueError, match="TIMESERIES"):
        MastDiscoveryClient(FixtureTransport([])).discover_time_series_named_target(
            "Vega",
            patch_id="series",
            radius_deg=0.1,
            filters=DiscoveryFilters(product_types=("IMAGE",)),
        )


def test_missing_optional_metadata_is_preserved_as_missing_and_no_uri_is_downloaded() -> None:
    observation = {
        "obsid": "minimal",
        "dataRights": "public",
        "dataproduct_type": "TIMESERIES",
        "t_min": 61000.0,
        "t_max": 61000.1,
    }
    product = {
        "obsid": "minimal",
        "dataRights": "public",
        "dataURI": "mast:HST/product/minimal.fits",
    }
    transport = FixtureTransport([{"data": [observation]}, {"data": [product]}])

    manifest = MastDiscoveryClient(transport).discover_sky_patch(
        patch_id="minimal", ra_deg=1.0, dec_deg=2.0, radius_deg=0.01
    )

    record = manifest.records[0]
    assert record.product_uri == "mast:HST/product/minimal.fits"
    assert record.download_uri is None
    assert record.wavelength.minimum_nm is None
    assert record.wavelength.maximum_nm is None
    assert record.wavelength.passband is None
    assert record.exposure_duration_seconds is None
    assert record.wcs_uri is None
    assert len(transport.calls) == 2
    assert all(call[1]["service"] in {"Mast.Caom.Cone", "Mast.Caom.Products"} for call in transport.calls)


def test_mast_json_transport_posts_request_without_credentials() -> None:
    requests: list[tuple[Any, float]] = []

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def read(self) -> bytes:
            return b'{"data": []}'

    def opener(request: Any, timeout: float) -> Response:
        requests.append((request, timeout))
        return Response()

    result = MastJsonTransport(
        endpoint="https://example.invalid/api/v0/invoke", opener=opener, timeout=7.0
    ).post_json("https://example.invalid/api/v0/invoke", {"service": "Test.Service"})

    assert result == {"data": []}
    request, timeout = requests[0]
    body = parse_qs(request.data.decode("utf-8"))
    assert json.loads(body["request"][0]) == {"service": "Test.Service"}
    assert request.headers["Content-type"] == "application/x-www-form-urlencoded"
    assert "authorization" not in {key.lower() for key in request.headers}
    assert timeout == 7.0


def test_invalid_name_lookup_response_raises_actionable_error() -> None:
    transport = FixtureTransport([{"resolvedCoordinate": []}])

    with pytest.raises(MastResponseError, match="no coordinates"):
        MastDiscoveryClient(transport).discover_named_target(
            "not-a-star", patch_id="empty", radius_deg=0.1
        )
