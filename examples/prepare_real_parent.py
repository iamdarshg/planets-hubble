"""Prepare downloaded HST FITS arrays for low-RSS repeated GPU workers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset.models import ManifestRecord, WavelengthMetadata  # noqa: E402
from dataset.parent_loader import FitsManifestParentLoader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = tuple(
        ManifestRecord(
            **{
                **item,
                "wavelength": WavelengthMetadata(**item.get("wavelength", {})),
            }
        )
        for item in source["records"]
    )
    paths = {item["product_id"]: args.manifest.parent / item["product_id"] for item in source["records"]}
    from astropy.time import Time

    loader = FitsManifestParentLoader(
        paths,
        target_shape=(720, 1280),
        time_converter_label="MJD_to_TDB_scale_only_local_probe",
        time_converter=lambda record: (
            Time(record.observation_start, format="mjd", scale="utc").tdb.jd,
            Time(record.observation_end, format="mjd", scale="utc").tdb.jd,
        ),
    )
    parent = loader.load(
        records,
        target_id=str(source["target"]),
        source_x=640.0,
        source_y=360.0,
        observation_id=args.manifest.parent.name,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    prepared_records = []
    for exposure, record in zip(parent.exposures, records):
        filename = f"{exposure.exposure_id}.npz"
        np.savez_compressed(
            args.output / filename,
            science=np.asarray(exposure.science),
            uncertainty=np.asarray(exposure.uncertainty),
            dq=np.asarray(exposure.dq),
        )
        prepared_records.append({"record": record.to_dict(), "array_file": filename})
    prepared = {
        "target": source["target"],
        "ra_deg": source["ra_deg"],
        "dec_deg": source["dec_deg"],
        "records": prepared_records,
        "source": "MAST",
        "preparation": "720x1280 centered crop/pad from downloaded HST FITS; cadence and product metadata preserved",
    }
    (args.output / "manifest.json").write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(prepared_records), "manifest": str(args.output / 'manifest.json')}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
