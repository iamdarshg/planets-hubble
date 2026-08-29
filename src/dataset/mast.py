"""MAST REST discovery, response normalization, and manifest assembly."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .manifest import build_manifest_record
from .models import DiscoveryManifest, ManifestRecord, SkyPosition
from .transport import JsonTransport


class MastResponseError(ValueError):
    """Raised when a MAST response cannot support a discovery operation."""


@dataclass(frozen=True)
class DiscoveryFilters:
    instruments: tuple[str, ...] = ()
    product_types: tuple[str, ...] = ()
    calibration_levels: tuple[int, ...] = ()
    passbands: tuple[str, ...] = ()


class MastDiscoveryClient:
    """Read-only client for named targets and explicit sky patches."""

    def __init__(
        self,
        transport: JsonTransport,
        *,
        endpoint: str = "https://mast.stsci.edu/api/v0/invoke",
    ) -> None:
        self.transport = transport
        self.endpoint = endpoint

    def discover_named_target(
        self,
        name: str,
        *,
        patch_id: str,
        radius_deg: float,
        filters: DiscoveryFilters | None = None,
        search_service: str = "Mast.Caom.Cone",
    ) -> DiscoveryManifest:
        response = self._request(
            "Mast.Name.Lookup", {"input": name, "format": "json"}, include_paging=False
        )
        position, source_identifier = _parse_name_lookup(response)
        return self._discover_position(
            patch_id=patch_id,
            position=position,
            radius_deg=radius_deg,
            filters=filters or DiscoveryFilters(),
            search_service=search_service,
            source_identifier=source_identifier,
        )

    def discover_sky_patch(
        self,
        *,
        patch_id: str,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
        filters: DiscoveryFilters | None = None,
        instruments: Sequence[str] = (),
        product_types: Sequence[str] = (),
        calibration_levels: Sequence[int] = (),
        passbands: Sequence[str] = (),
        search_service: str = "Mast.Caom.Cone",
    ) -> DiscoveryManifest:
        combined = filters or DiscoveryFilters(
            instruments=tuple(instruments),
            product_types=tuple(product_types),
            calibration_levels=tuple(calibration_levels),
            passbands=tuple(passbands),
        )
        if filters is not None and any(
            (instruments, product_types, calibration_levels, passbands)
        ):
            raise ValueError("pass filters or individual filter sequences, not both")
        return self._discover_position(
            patch_id=patch_id,
            position=SkyPosition(float(ra_deg), float(dec_deg)),
            radius_deg=radius_deg,
            filters=combined,
            search_service=search_service,
            source_identifier=None,
        )

    def _discover_position(
        self,
        *,
        patch_id: str,
        position: SkyPosition,
        radius_deg: float,
        filters: DiscoveryFilters,
        search_service: str,
        source_identifier: str | None,
    ) -> DiscoveryManifest:
        _validate_position(position, radius_deg)
        params: dict[str, Any]
        if search_service == "Mast.Caom.Cone":
            params = {
                "ra": position.ra_deg,
                "dec": position.dec_deg,
                "radius": radius_deg,
            }
        elif search_service == "Mast.Caom.Filtered.Position":
            params = {
                "position": f"{position.ra_deg}, {position.dec_deg}, {radius_deg}",
                "columns": "*",
                "filters": _mast_filters(filters),
            }
        else:
            raise ValueError(
                "search_service must be Mast.Caom.Cone or "
                "Mast.Caom.Filtered.Position"
            )

        response = self._request(search_service, params)
        observations = _rows(response, "observation search")
        records: list[ManifestRecord] = []
        for observation in observations:
            if not _is_public_science(observation, filters):
                continue
            obsid = _required_identifier(observation, "obsid", "observation search")
            product_response = self._request(
                "Mast.Caom.Products", {"obsid": obsid}
            )
            for product in _rows(product_response, f"products for {obsid}"):
                if _is_public_product(product):
                    records.append(build_manifest_record(observation, product))

        return DiscoveryManifest(
            patch_id=patch_id,
            target_position=position,
            source_identifier=source_identifier,
            radius_deg=float(radius_deg),
            search_service=search_service,
            records=tuple(records),
        )

    def _request(
        self,
        service: str,
        params: Mapping[str, Any],
        *,
        include_paging: bool = True,
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {"service": service, "params": dict(params), "format": "json"}
        if include_paging:
            payload.update({"pagesize": 2000, "page": 1, "removenullcolumns": True})
        response = self.transport.post_json(self.endpoint, payload)
        if response.get("status") == "ERROR":
            raise MastResponseError(f"MAST {service} failed: {response.get('msg', 'unknown error')}")
        return response


def _parse_name_lookup(response: Mapping[str, Any]) -> tuple[SkyPosition, str | None]:
    candidates = response.get("resolvedCoordinate")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise MastResponseError("name lookup returned no coordinates")
    first = candidates[0]
    if not isinstance(first, Mapping):
        raise MastResponseError("name lookup coordinate was not an object")
    try:
        declination = first.get("decl")
        if declination is None:
            declination = first["dec"]
        position = SkyPosition(float(first["ra"]), float(declination))
    except (KeyError, TypeError, ValueError) as exc:
        raise MastResponseError("name lookup coordinate lacks numeric ra/dec") from exc
    canonical = first.get("canonicalName")
    return position, str(canonical) if canonical is not None else None


def _rows(response: Mapping[str, Any], operation: str) -> list[Mapping[str, Any]]:
    rows = response.get("data", [])
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise MastResponseError(f"MAST {operation} returned invalid data rows")
    return rows


def _required_identifier(row: Mapping[str, Any], key: str, operation: str) -> str:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        raise MastResponseError(f"MAST {operation} row lacks {key}")
    return str(value)


def _is_public_science(row: Mapping[str, Any], filters: DiscoveryFilters) -> bool:
    rights = str(row.get("dataRights", "")).lower()
    if rights != "public":
        return False
    if filters.instruments and str(row.get("instrument_name", "")) not in filters.instruments:
        return False
    if filters.product_types and str(row.get("dataproduct_type", "")).upper() not in {
        value.upper() for value in filters.product_types
    }:
        return False
    if filters.calibration_levels and _int_or_none(row.get("calib_level")) not in filters.calibration_levels:
        return False
    if filters.passbands and not _contains_any(row.get("filters"), filters.passbands):
        return False
    return True


def _is_public_product(row: Mapping[str, Any]) -> bool:
    return str(row.get("dataRights", "")).lower() == "public"


def _contains_any(value: Any, choices: Iterable[str]) -> bool:
    normalized = {part.strip().upper() for part in str(value or "").split(";")}
    return any(str(choice).upper() in normalized for choice in choices)


def _mast_filters(filters: DiscoveryFilters) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = [
        {"paramName": "dataRights", "values": ["public"]},
    ]
    for name, values in (
        ("instrument_name", filters.instruments),
        ("dataproduct_type", filters.product_types),
        ("calib_level", filters.calibration_levels),
        ("filters", filters.passbands),
    ):
        if values:
            result.append({"paramName": name, "values": list(values)})
    return result


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_position(position: SkyPosition, radius_deg: float) -> None:
    if not 0.0 <= position.ra_deg < 360.0:
        raise ValueError("ra_deg must be in [0, 360)")
    if not -90.0 <= position.dec_deg <= 90.0:
        raise ValueError("dec_deg must be in [-90, 90]")
    if radius_deg <= 0.0:
        raise ValueError("radius_deg must be positive")
