"""Build a tiny MAST-shaped manifest without contacting the network.

Run from the repository root with::

    python examples/dataset_manifest_fixture.py

The fixture transport records the same requests that the production client
would send, while returning small in-memory responses.  This makes the
example safe for CI and useful for inspecting the normalized manifest shape.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset import MastDiscoveryClient  # noqa: E402


class FixtureTransport:
    """Minimal in-memory implementation of the dataset transport protocol."""

    def __init__(self, responses: list[Mapping[str, Any]]) -> None:
        self._responses = list(responses)

    def post_json(self, url: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        del url, payload
        if not self._responses:
            raise RuntimeError("fixture received an unexpected extra request")
        return self._responses.pop(0)


def build_fixture_manifest() -> dict[str, Any]:
    """Return one normalized manifest record from MAST-shaped fixture rows."""

    observation = {
        "obsid": "hst-fixture-001",
        "dataRights": "public",
        "instrument_name": "WFC3/UVIS",
        "dataproduct_type": "IMAGE",
        "calib_level": 3,
        "t_min": 60000.0,
        "t_max": 60000.25,
        "t_exptime": 900.0,
        "s_ra": 279.2347,
        "s_dec": 38.7837,
        "s_region": "Circle ICRS 279.2347 38.7837 0.01",
        "filters": "F275W",
    }
    product = {
        "obsid": "hst-fixture-001",
        "obs_id": "hst-fixture-001-drz",
        "dataRights": "public",
        "dataURI": "mast:HST/product/hst-fixture-001_drz.fits",
        "dataURL": "https://mast.stsci.edu/download/fixture",
        "productType": "SCIENCE",
        "em_min": 250.0,
        "em_max": 900.0,
    }
    transport = FixtureTransport([{"data": [observation]}, {"data": [product]}])
    manifest = MastDiscoveryClient(transport).discover_sky_patch(
        patch_id="fixture-patch",
        ra_deg=279.2347,
        dec_deg=38.7837,
        radius_deg=0.01,
    )
    return manifest.to_dict()


if __name__ == "__main__":
    print(json.dumps(build_fixture_manifest(), indent=2, sort_keys=True))
