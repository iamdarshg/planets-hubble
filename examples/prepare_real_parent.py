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


def _source_pixel_in_patch(
    path: Path,
    *,
    ra_deg: float,
    dec_deg: float,
    target_shape: tuple[int, int] = (720, 1280),
) -> tuple[float, float, str]:
    """Map the target sky position into the same crop/pad as the FITS arrays.

    Prepared parents must retain the target's actual detector position.  A
    fixed crop center is not a valid source anchor: HST pointings can place a
    target away from the detector center, and injecting there would create a
    learnable ``empty-patch`` shortcut.
    """

    try:
        from astropy.coordinates import SkyCoord
        from astropy.io import fits
        from astropy.wcs import WCS
        import astropy.units as u

        with fits.open(path, memmap=True) as hdul:
            sci = next(
                (hdu for hdu in hdul if str(getattr(hdu, "name", "")).upper() == "SCI"),
                hdul[0],
            )
            if sci.data is None or np.asarray(sci.data).ndim != 2:
                return target_shape[1] / 2.0, target_shape[0] / 2.0, "fallback_no_2d_sci"
            height, width = np.asarray(sci.data).shape
            pixel_x, pixel_y = WCS(sci.header).world_to_pixel(
                SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
            )

        target_height, target_width = target_shape
        crop_y = max((height - target_height) // 2, 0)
        crop_x = max((width - target_width) // 2, 0)
        pad_y = max((target_height - height) // 2, 0)
        pad_x = max((target_width - width) // 2, 0)
        patched_x = float(pixel_x) - crop_x + pad_x
        patched_y = float(pixel_y) - crop_y + pad_y
        if not np.isfinite(patched_x) or not np.isfinite(patched_y):
            raise ValueError("WCS returned a non-finite pixel")
        if not (0.0 <= patched_x < target_width and 0.0 <= patched_y < target_height):
            raise ValueError("target is outside the prepared patch")
        return patched_x, patched_y, "SCI_WCS_world_to_pixel"
    except Exception as exc:
        # A prepared dataset remains usable when a product has incomplete WCS,
        # but the provenance records that the center fallback was used.
        print(f"warning: WCS source mapping failed for {path.name}: {exc}")
        return target_shape[1] / 2.0, target_shape[0] / 2.0, "fallback_center"


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
    first_product = args.manifest.parent / records[0].product_id
    source_x, source_y, source_position_method = _source_pixel_in_patch(
        first_product,
        ra_deg=float(source["ra_deg"]),
        dec_deg=float(source["dec_deg"]),
    )
    parent = loader.load(
        records,
        target_id=str(source["target"]),
        source_x=source_x,
        source_y=source_y,
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
        "source_x": source_x,
        "source_y": source_y,
        "source_position_method": source_position_method,
        "records": prepared_records,
        "source": "MAST",
        "preparation": "720x1280 centered crop/pad from downloaded HST FITS; target source mapped with SCI WCS when available; cadence and product metadata preserved",
    }
    (args.output / "manifest.json").write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(prepared_records), "manifest": str(args.output / 'manifest.json')}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
