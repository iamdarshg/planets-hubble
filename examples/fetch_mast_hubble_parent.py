"""Fetch a small, external real-HST parent set through the MAST REST API.

The downloaded FITS products and generated manifest are intentionally kept
outside Git.  This example selects public WFC3/IR science products for a named
known exoplanet host, writes only a compact JSON manifest, and enforces a byte
budget before downloading.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


ENDPOINT = "https://mast.stsci.edu/api/v0/invoke"
DOWNLOAD_ENDPOINT = "https://mast.stsci.edu/api/v0.1/Download/file"


def invoke(service: str, params: dict[str, object], *, pagesize: int = 2000) -> dict[str, object]:
    payload = {
        "service": service,
        "params": params,
        "format": "json",
        "pagesize": pagesize,
        "page": 1,
        "removenullcolumns": True,
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=urllib.parse.urlencode({"request": json.dumps(payload)}).encode(),
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        result = json.loads(response.read())
    if result.get("status") == "ERROR":
        raise RuntimeError(f"MAST {service} failed: {result.get('msg', 'unknown error')}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="HD 209458")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-bytes", type=int, default=200_000_000)
    parser.add_argument("--output", type=Path, default=Path("data/real/hd209458"))
    args = parser.parse_args()
    if args.count < 1 or args.offset < 0 or args.max_bytes < 1:
        raise ValueError("count/max-bytes must be positive and offset non-negative")

    resolved = invoke("Mast.Name.Lookup", {"input": args.name, "format": "json"}, pagesize=1)
    coordinates = resolved.get("resolvedCoordinate", [])
    if not coordinates:
        raise RuntimeError(f"MAST could not resolve {args.name!r}")
    position = coordinates[0]
    ra = float(position["ra"])
    dec = float(position.get("decl", position.get("dec")))
    search = invoke(
        "Mast.Caom.Filtered.Position",
        {
            "position": f"{ra}, {dec}, 0.02",
            "columns": "*",
            "filters": [
                {"paramName": "dataRights", "values": ["public"]},
                {"paramName": "instrument_name", "values": ["WFC3/IR"]},
            ],
        },
    )
    normalized_name = "".join(character for character in args.name.lower() if character.isalnum())
    rows = [
        row
        for row in search.get("data", [])
        if str(row.get("intentType", "")).lower() == "science"
        and str(row.get("instrument_name", "")) == "WFC3/IR"
        and normalized_name in "".join(character for character in str(row.get("target_name", "")).lower() if character.isalnum())
        and row.get("obsid") is not None
        and row.get("t_min") is not None
        and row.get("t_max") is not None
    ]
    if not rows:
        raise RuntimeError("MAST returned no public WFC3/IR science rows for the target")

    proposal = Counter(str(row.get("proposal_id", "unknown")) for row in rows).most_common(1)[0][0]
    rows = sorted(
        (row for row in rows if str(row.get("proposal_id", "unknown")) == proposal),
        key=lambda row: float(row["t_min"]),
    )[args.offset : args.offset + args.count]

    records: list[dict[str, object]] = []
    downloads: list[tuple[str, str, int]] = []
    for row in rows:
        products = invoke("Mast.Caom.Products", {"obsid": str(row["obsid"])})
        choices = [
            product
            for product in products.get("data", [])
            if str(product.get("dataRights", "")).lower() == "public"
            and str(product.get("productType", "")).upper() == "SCIENCE"
            and str(product.get("productSubGroupDescription", "")).upper() in {"FLT", "IMA"}
            and product.get("dataURI")
        ]
        if not choices:
            continue
        product = sorted(
            choices,
            key=lambda item: (str(item.get("productSubGroupDescription", "")).upper() != "FLT", int(item.get("size", 0) or 0)),
        )[0]
        product_uri = str(product["dataURI"])
        filename = str(product.get("productFilename") or Path(urllib.parse.urlparse(product_uri).path).name)
        size = int(product.get("size", 0) or 0)
        downloads.append((filename, product_uri, size))
        records.append(
            {
                "observation_id": str(row["obsid"]),
                "product_id": filename,
                "product_uri": product_uri,
                "download_uri": f"{DOWNLOAD_ENDPOINT}?{urllib.parse.urlencode({'uri': product_uri})}",
                "observation_start": row["t_min"],
                "observation_midpoint": (float(row["t_min"]) + float(row["t_max"])) / 2.0,
                "observation_end": row["t_max"],
                "time_system": "MJD",
                "exposure_duration_seconds": row.get("t_exptime"),
                "wavelength": {
                    "minimum_nm": row.get("em_min"),
                    "maximum_nm": row.get("em_max"),
                    "passband": row.get("filters"),
                },
                "calibration_level": product.get("calib_level", row.get("calib_level")),
                "product_type": row.get("dataproduct_type"),
                "instrument": row.get("instrument_name"),
                "spatial_footprint": {
                    "detector": "IR",
                    "s_fov": row.get("s_fov"),
                    "s_pixel_scale": row.get("s_pixel_scale"),
                },
                "coverage": {"proposal_id": proposal},
                "quality": {},
            }
        )

    total = sum(size for _, _, size in downloads)
    if not downloads or total > args.max_bytes:
        raise RuntimeError(f"selected FITS size {total} exceeds max-bytes {args.max_bytes}")
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, product_uri, _ in downloads:
        destination = args.output / filename
        if not destination.exists():
            url = f"{DOWNLOAD_ENDPOINT}?{urllib.parse.urlencode({'uri': product_uri})}"
            urllib.request.urlretrieve(url, destination)
    manifest = {
        "target": args.name,
        "ra_deg": ra,
        "dec_deg": dec,
        "records": records,
        "downloaded_bytes_declared": total,
        "source": "MAST",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "declared_bytes": total, "manifest": str(args.output / 'manifest.json')}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
