"""Normalize MAST observation/product rows into manifest records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .models import ManifestRecord, WavelengthMetadata


def build_manifest_record(
    observation: Mapping[str, Any], product: Mapping[str, Any]
) -> ManifestRecord:
    """Combine a CAOM observation row with one of its product rows."""

    start = _first(observation, product, "t_min", "start_time")
    end = _first(observation, product, "t_max", "end_time")
    wavelength = WavelengthMetadata(
        minimum_nm=_float_or_none(_first(product, observation, "em_min")),
        maximum_nm=_float_or_none(_first(product, observation, "em_max")),
        passband=_string_or_none(_first(product, observation, "filters", "passband")),
    )
    return ManifestRecord(
        observation_id=_required_identifier(observation, "obsid", "observation"),
        product_id=_string_or_none(_first(product, product, "obs_id", "productFilename")),
        product_uri=_string_or_none(_first(product, product, "dataURI", "productURI")),
        download_uri=_string_or_none(_first(product, product, "dataURL", "downloadURI")),
        observation_start=start,
        observation_midpoint=_midpoint(start, end),
        observation_end=end,
        time_system=_time_system(observation, product, start, end),
        exposure_duration_seconds=_float_or_none(
            _first(product, observation, "t_exptime", "exposure_duration")
        ),
        wavelength=wavelength,
        calibration_level=_int_or_none(_first(product, observation, "calib_level")),
        product_type=_string_or_none(
            _first(product, observation, "dataproduct_type", "productType")
        ),
        instrument=_string_or_none(_first(observation, product, "instrument_name", "instrument")),
        wcs_uri=_string_or_none(_first(product, observation, "wcs_uri", "wcsURI")),
        spatial_footprint=_known_mapping(
            observation,
            ("s_ra", "s_dec", "s_region", "s_fov", "s_pixel_scale"),
        ),
        observer_position=_mapping_or_none(
            _first(product, observation, "observer_position", "r_geo")
        ),
        observer_velocity=_mapping_or_none(
            _first(product, observation, "observer_velocity", "v_geo")
        ),
        pointing=_merged_known_mapping(
            product,
            observation,
            (
                "pointing",
                "roll_deg",
                "off_axis_angle_deg",
                "boresight_ra_deg",
                "boresight_dec_deg",
                "solar_elongation_deg",
                "pa_aper",
            ),
        ),
        coverage=_merged_known_mapping(
            product, observation, ("exposure_count", "coverage", "s_region", "valid_fraction")
        ),
        quality=_merged_known_mapping(
            product, observation, ("quality", "quality_flag", "data_quality")
        ),
    )


def _first(primary: Mapping[str, Any], secondary: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in primary and primary[key] is not None:
            return primary[key]
        if key in secondary and secondary[key] is not None:
            return secondary[key]
    return None


def _midpoint(start: Any, end: Any) -> Any:
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return (float(start) + float(end)) / 2.0
    return None


def _time_system(
    observation: Mapping[str, Any], product: Mapping[str, Any], start: Any, end: Any
) -> str | None:
    declared = _first(observation, product, "time_system", "timeSystem")
    if declared is not None:
        return str(declared)
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return "MJD"
    return None


def _known_mapping(row: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any] | None:
    result = {key: row[key] for key in keys if key in row and row[key] is not None}
    return result or None


def _merged_known_mapping(
    primary: Mapping[str, Any], secondary: Mapping[str, Any], keys: Iterable[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in (secondary, primary):
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            if isinstance(value, Mapping):
                result.update(value)
            else:
                result[key] = value
    return result


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _required_identifier(row: Mapping[str, Any], key: str, operation: str) -> str:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"MAST {operation} row lacks {key}")
    return str(value)
