"""Join Arcsecond and optional NASA ADS metadata to a real-data manifest.

This is a provenance/enrichment step, not a label generator.  Arcsecond's
public exoplanet endpoint provides catalogue coordinates, orbital parameters,
stellar properties, discovery metadata, and source links.  Its SIMBAD-backed
object endpoint provides aliases/object types/flux metadata.  The NASA ADS
Developer API is optional because it requires a user token.

The output is a sidecar JSONL keyed by ``target_name``.  The original manifest,
NPZ files, labels, and split assignments are never modified.  Positions are
reported as propagated catalogue sky positions and line-of-sight vectors; the
catalogue does not provide a resolved three-dimensional planet position from
Earth.  Orbital phase is therefore included separately from the host-system
sky position.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin
from urllib.request import Request, urlopen


ARCSECOND_BASE = "https://api.arcsecond.io/"
ADS_ENDPOINT = "https://api.adsabs.harvard.edu/v1/search/query"
J2000_JD = 2451545.0
MAS_PER_DEGREE = 3_600_000.0
AU_PER_PC = 206_264.80624709636


class JsonHttpError(RuntimeError):
    """A bounded, non-secret HTTP failure."""


def _request_json(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30.0,
    retries: int = 2,
) -> tuple[int, Any]:
    request_headers = {"User-Agent": "planets-hubble-metadata/1.0"}
    if headers:
        request_headers.update(headers)
    last_error: Exception | None = None
    retry_delay = 0.0
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers=request_headers, method="GET")
            with urlopen(request, timeout=timeout) as response:
                status = int(response.status)
                payload = json.loads(response.read().decode("utf-8"))
            return status, payload
        except HTTPError as exc:
            # Do not include response bodies: an auth error can contain
            # service-specific details and never needs to be persisted here.
            if exc.code in {400, 401, 403, 404, 422}:
                raise JsonHttpError(f"HTTP {exc.code} for {url}") from exc
            last_error = exc
            if exc.code == 429:
                try:
                    retry_delay = max(float(exc.headers.get("Retry-After", "0")), 5.0)
                except (TypeError, ValueError):
                    retry_delay = 5.0 * (attempt + 1)
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < retries:
            delay = retry_delay if retry_delay > 0.0 else 0.5 * (attempt + 1)
            time.sleep(delay)
    raise JsonHttpError(f"request failed for {url}: {last_error}") from last_error


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"manifest row is not an object: {path}")
                rows.append(value)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _target_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = str(row.get("target_name") or "").strip()
        if not target:
            continue
        group = groups.setdefault(
            target,
            {
                "target_name": target,
                "example_ids": [],
                "splits": [],
                "labels": [],
                "observation_epochs_jd": [],
                "manifest_coordinates": None,
            },
        )
        group["example_ids"].append(str(row.get("example_id", "")))
        group["splits"].append(str(row.get("split", "")))
        group["labels"].append(int(row.get("label", 0)))
        bkjd = _float(row.get("center_bkjd"))
        if bkjd is not None:
            group["observation_epochs_jd"].append(bkjd + 2_454_833.0)
        tpf = row.get("provenance", {}).get("tpf_metadata", {})
        if group["manifest_coordinates"] is None and isinstance(tpf, Mapping):
            ra = _float(tpf.get("ra"))
            dec = _float(tpf.get("dec"))
            if ra is not None and dec is not None:
                group["manifest_coordinates"] = {"right_ascension": ra, "declination": dec}
    return groups


def _safe_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cache_path(cache_dir: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{digest}.json"


def _get_cached_json(
    url: str,
    *,
    cache_dir: Path,
    headers: Mapping[str, str] | None = None,
    refresh: bool = False,
) -> tuple[int, Any, bool]:
    path = _cache_path(cache_dir, url)
    if path.exists() and not refresh:
        return 200, json.loads(path.read_text(encoding="utf-8")), True
    status, payload = _request_json(url, headers=headers)
    _safe_json_write(path, payload)
    return status, payload, False


def _fetch_arcsecond_exoplanets(
    *,
    base_url: str,
    cache_dir: Path,
    refresh: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    page_url = urljoin(base_url.rstrip("/") + "/", "exoplanets/?page=1")
    results: list[dict[str, Any]] = []
    pages = 0
    cache_hits = 0
    seen_urls: set[str] = set()
    while page_url and page_url not in seen_urls:
        seen_urls.add(page_url)
        _, payload, cached = _get_cached_json(
            page_url,
            cache_dir=cache_dir / "exoplanets",
            refresh=refresh,
        )
        if not isinstance(payload, Mapping) or not isinstance(payload.get("results"), list):
            raise JsonHttpError(f"Arcsecond response did not contain results: {page_url}")
        pages += 1
        cache_hits += int(cached)
        results.extend(item for item in payload["results"] if isinstance(item, dict))
        next_url = payload.get("next")
        page_url = str(next_url) if next_url else ""
    return results, {
        "pages": pages,
        "records": len(results),
        "cache_hits": cache_hits,
        "endpoint": urljoin(base_url.rstrip("/") + "/", "exoplanets/"),
    }


def _name_variants(record: Mapping[str, Any]) -> set[str]:
    values = {
        str(record.get("name") or "").strip().casefold(),
        str(record.get("pl_name") or "").strip().casefold(),
    }
    host = str(record.get("hostname") or "").strip()
    letter = str(record.get("pl_letter") or "").strip()
    if host and letter:
        values.add(f"{host} {letter}".casefold())
    return {value for value in values if value}


def _match_exoplanet(records: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    target_key = target.strip().casefold()
    exact = [record for record in records if target_key in _name_variants(record)]
    if exact:
        return exact[0]
    compact = target_key.replace(" ", "")
    for record in records:
        if any(compact == value.replace(" ", "") for value in _name_variants(record)):
            return record
    return None


def _selected_exoplanet_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "id", "name", "pl_name", "hostname", "parent_star", "source", "date_sync",
        "hd_name", "hip_name", "tic_id", "gaia_dr2_id", "gaia_dr3_id",
        "discoverymethod", "disc_year", "disc_refname", "disc_locale", "disc_facility",
        "disc_telescope", "disc_instrument", "equatorial_coordinates",
        "pl_orbper", "pl_orbpererr1", "pl_orbpererr2", "pl_orbsmax", "pl_orbeccen",
        "pl_orbincl", "pl_orbtper", "pl_orblper", "pl_tranmid", "pl_tranmiderr1",
        "pl_tranmiderr2", "pl_imppar", "pl_trandep", "pl_trandur", "pl_ratdor", "pl_ratror",
        "st_spectype", "st_teff", "st_rad", "st_mass", "st_met", "st_logg", "st_dens",
        "sy_dist", "sy_disterr1", "sy_disterr2", "sy_plx", "sy_plxerr1", "sy_plxerr2",
        "sy_pm", "sy_pmra", "sy_pmdec", "sy_bmag", "sy_vmag",
    )
    return {field: record.get(field) for field in fields if field in record}


def _propagated_position(
    record: Mapping[str, Any],
    observation_epochs_jd: list[float],
) -> dict[str, Any]:
    coordinates = record.get("equatorial_coordinates")
    if not isinstance(coordinates, Mapping):
        return {"status": "missing_catalogue_coordinates", "epochs": []}
    ra = _float(coordinates.get("right_ascension"))
    dec = _float(coordinates.get("declination"))
    epoch = _float(coordinates.get("epoch")) or J2000_JD
    if ra is None or dec is None:
        return {"status": "invalid_catalogue_coordinates", "epochs": []}

    distance_pc = _float(record.get("sy_dist"))
    if distance_pc is None:
        parallax_mas = _float(record.get("sy_plx"))
        if parallax_mas is not None and parallax_mas > 0.0:
            distance_pc = 1000.0 / parallax_mas
    pmra = _float(record.get("sy_pmra")) or 0.0
    pmdec = _float(record.get("sy_pmdec")) or 0.0
    period_days = _float(record.get("pl_orbper"))
    transit_mid_jd = _float(record.get("pl_tranmid"))
    epochs: list[dict[str, Any]] = []
    for jd in sorted(set(observation_epochs_jd)):
        years = (jd - epoch) / 365.25
        dec_at_epoch = dec + (pmdec / MAS_PER_DEGREE) * years
        cos_dec = max(abs(math.cos(math.radians(dec_at_epoch))), 1.0e-8)
        ra_at_epoch = (ra + (pmra / MAS_PER_DEGREE) * years / cos_dec) % 360.0
        ra_rad = math.radians(ra_at_epoch)
        dec_rad = math.radians(dec_at_epoch)
        unit = [
            math.cos(dec_rad) * math.cos(ra_rad),
            math.cos(dec_rad) * math.sin(ra_rad),
            math.sin(dec_rad),
        ]
        entry: dict[str, Any] = {
            "jd": jd,
            "catalogue_epoch_jd": epoch,
            "years_from_catalogue_epoch": years,
            "right_ascension_deg": ra_at_epoch,
            "declination_deg": dec_at_epoch,
            "line_of_sight_unit_heliocentric_equatorial": unit,
            "position_status": "catalogue_propagated_sky_position",
        }
        if distance_pc is not None and distance_pc > 0.0:
            entry["distance_pc"] = distance_pc
            entry["line_of_sight_vector_au"] = [value * distance_pc * AU_PER_PC for value in unit]
        if transit_mid_jd is not None and period_days is not None and period_days > 0.0:
            cycles = (jd - transit_mid_jd) / period_days
            entry["orbital_cycles_from_transit_midpoint"] = cycles
            entry["orbital_phase"] = cycles % 1.0
            entry["transit_midpoint_offset_days"] = (cycles - round(cycles)) * period_days
        epochs.append(entry)
    return {
        "status": "complete",
        "coordinate_epoch_jd": epoch,
        "distance_source": "sy_dist_pc" if _float(record.get("sy_dist")) is not None else "inverse_sy_plx_mas",
        "proper_motion_source": "sy_pmra_sy_pmdec_mas_per_year",
        "epochs": epochs,
    }


def _object_url(base_url: str, target: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", f"objects/{quote(target, safe='')}/")


def _fetch_object(
    target: str,
    *,
    base_url: str,
    cache_dir: Path,
    refresh: bool,
) -> tuple[str, dict[str, Any] | None, str | None, bool]:
    url = _object_url(base_url, target)
    try:
        _, payload, cached = _get_cached_json(url, cache_dir=cache_dir / "objects", refresh=refresh)
        if isinstance(payload, dict):
            selected = {
                key: payload.get(key)
                for key in (
                    "name", "source", "equatorial_coordinates", "ecliptic_coordinates",
                    "galactic_coordinates", "proper_motion", "parallax", "radial_velocity",
                    "distance", "metallicity", "effective_temperature", "aliases",
                    "object_types", "fluxes",
                )
                if key in payload
            }
            return target, selected, None, cached
        return target, None, "response was not an object", cached
    except JsonHttpError as exc:
        return target, None, str(exc), False


def _ads_query(
    target: str,
    *,
    token: str,
    rows: int,
    cache_dir: Path,
    refresh: bool,
    timeout: float = 30.0,
) -> tuple[str, dict[str, Any] | None, str | None]:
    # The ADS web search documents an object: modifier, but the current
    # Developer API deployment returns HTTP 400/"undefined field object" for
    # that modifier.  Abstract/title phrase search is supported by the API and
    # still discovers papers that name this target without pretending that a
    # failed object-field query is a successful association.
    query = f'abs:"{target}" OR title:"{target}"'
    params = urlencode(
        {
            "q": query,
            "rows": str(rows),
            "fl": "bibcode,title,year,author,facility,data,property,esources",
        }
    )
    url = f"{ADS_ENDPOINT}?{params}"
    try:
        _, payload, _ = _get_cached_json(
            url,
            cache_dir=cache_dir / "ads",
            headers={"Authorization": f"Bearer {token}"},
            refresh=refresh,
        )
        return target, payload if isinstance(payload, dict) else None, None
    except JsonHttpError as exc:
        return target, None, str(exc)


def _record_for_target(
    group: dict[str, Any],
    arcsecond_records: list[dict[str, Any]],
    *,
    object_data: dict[str, Any] | None,
    object_error: str | None,
    object_cached: bool,
    ads_data: dict[str, Any] | None,
    ads_error: str | None,
) -> dict[str, Any]:
    target = str(group["target_name"])
    arcsecond = _match_exoplanet(arcsecond_records, target)
    output: dict[str, Any] = {
        "target_name": target,
        "manifest": {
            "example_ids": group["example_ids"],
            "splits": sorted(set(group["splits"])),
            "labels": dict(Counter(group["labels"])),
            "observation_epochs_jd": sorted(set(group["observation_epochs_jd"])),
            "manifest_coordinates": group["manifest_coordinates"],
        },
        "sources": {
            "arcsecond": {
                "endpoint": f"{ARCSECOND_BASE}exoplanets/",
                "object_endpoint": _object_url(ARCSECOND_BASE, target),
                "status": "matched" if arcsecond is not None else "not_found",
            },
            "ads": {
                "endpoint": ADS_ENDPOINT,
                "status": "matched" if ads_data is not None else ("error" if ads_error else "not_requested"),
            },
        },
    }
    if arcsecond is not None:
        output["arcsecond_exoplanet"] = _selected_exoplanet_fields(arcsecond)
        output["earth_relative_position"] = _propagated_position(
            arcsecond,
            [float(value) for value in group["observation_epochs_jd"]],
        )
    else:
        output["earth_relative_position"] = {
            "status": "unavailable_without_arcsecond_match",
            "epochs": [],
        }
    output["arcsecond_simbad_object"] = object_data
    output["sources"]["arcsecond"]["object_status"] = (
        "matched" if object_data is not None else ("error" if object_error else "not_requested")
    )
    if object_error:
        output["sources"]["arcsecond"]["object_error"] = object_error
    output["sources"]["arcsecond"]["object_cache_hit"] = object_cached
    output["ads_records"] = ads_data.get("response", {}).get("docs", []) if ads_data else []
    if ads_error:
        output["sources"]["ads"]["error"] = ads_error
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arcsecond-base-url", default=ARCSECOND_BASE)
    parser.add_argument("--ads-token-env", default="ADS_API_TOKEN")
    parser.add_argument(
        "--ads-token-file",
        type=Path,
        default=Path.home() / ".ads" / "token",
        help="optional ADS token file, used only when the environment variable is absent",
    )
    parser.add_argument("--ads-rows", type=int, default=5)
    parser.add_argument("--ads-max-targets", type=int, default=0)
    parser.add_argument(
        "--object-max-targets",
        type=int,
        default=128,
        help="maximum number of slower SIMBAD-backed Arcsecond object lookups (0 disables them)",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.ads_rows < 1 or args.ads_max_targets < 0 or args.object_max_targets < 0:
        raise ValueError("workers and ads_rows must be positive; max-target values cannot be negative")

    rows = _load_manifest(args.manifest)
    groups = _target_groups(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "raw"
    arcsecond_records, arcsecond_fetch = _fetch_arcsecond_exoplanets(
        base_url=args.arcsecond_base_url,
        cache_dir=cache_dir,
        refresh=args.refresh,
    )

    object_targets = sorted(groups)
    if args.object_max_targets:
        object_targets = object_targets[: args.object_max_targets]
    object_results: dict[str, tuple[dict[str, Any] | None, str | None, bool]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _fetch_object,
                target,
                base_url=args.arcsecond_base_url,
                cache_dir=cache_dir,
                refresh=args.refresh,
            ): target
            for target in object_targets
        }
        for future in as_completed(futures):
            target, data, error, cached = future.result()
            object_results[target] = (data, error, cached)

    token = os.environ.get(args.ads_token_env, "").strip()
    token_source = "environment" if token else None
    if not token and args.ads_token_file.is_file():
        token = args.ads_token_file.read_text(encoding="utf-8").strip()
        token_source = "file" if token else None
    ads_targets = sorted(groups)
    if args.ads_max_targets:
        ads_targets = ads_targets[: args.ads_max_targets]
    ads_results: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
    if token:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _ads_query,
                    target,
                    token=token,
                    rows=args.ads_rows,
                    cache_dir=cache_dir,
                    refresh=args.refresh,
                ): target
                for target in ads_targets
            }
            for future in as_completed(futures):
                target, data, error = future.result()
                ads_results[target] = (data, error)
    else:
        for target in ads_targets:
            ads_results[target] = (None, None)

    output_path = args.output_dir / "target_context.jsonl"
    matched = 0
    object_ok = 0
    ads_docs = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for target in sorted(groups):
            object_data, object_error, object_cached = object_results.get(target, (None, "not returned", False))
            ads_data, ads_error = ads_results.get(target, (None, None))
            record = _record_for_target(
                groups[target],
                arcsecond_records,
                object_data=object_data,
                object_error=object_error,
                object_cached=object_cached,
                ads_data=ads_data,
                ads_error=ads_error,
            )
            matched += int(record["sources"]["arcsecond"]["status"] == "matched")
            object_ok += int(object_data is not None)
            ads_docs += len(record["ads_records"])
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    report = {
        "status": "complete",
        "manifest": str(args.manifest),
        "output": str(output_path),
        "targets": len(groups),
        "manifest_examples": len(rows),
        "arcsecond": {
            **arcsecond_fetch,
            "matched_targets": matched,
            "object_records": object_ok,
            "object_targets_requested": len(object_targets),
            "object_endpoint": f"{args.arcsecond_base_url.rstrip('/')}/objects/{{target}}/",
        },
        "ads": {
            "endpoint": ADS_ENDPOINT,
            "token_env": args.ads_token_env,
            "token_file": str(args.ads_token_file),
            "token_source": token_source,
            "authenticated": bool(token),
            "queried_targets": len(ads_targets) if token else 0,
            "records_returned": ads_docs,
            "status": "complete" if token else "blocked_authentication_required",
        },
        "position_semantics": {
            "sky_position": "Arcsecond catalogue RA/DEC propagated with catalogue proper motion to each manifest centre epoch",
            "line_of_sight_vector": "heliocentric equatorial vector in AU using catalogue distance/parallax; not a resolved planet barycentric position",
            "orbital_phase": "phase from Arcsecond transit midpoint and orbital period when both are available",
            "earth_observer_correction": "not applied; the sidecar retains the catalogue/heliocentric approximation and does not invent an Earth ephemeris",
        },
        "label_policy": "sidecar only; original manifest labels, splits, and NPZ science arrays are unchanged",
    }
    _safe_json_write(args.output_dir / "report.json", report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
