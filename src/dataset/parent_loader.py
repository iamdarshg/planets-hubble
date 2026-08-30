"""Build immutable real-observation parents from MAST manifest records.

The discovery client deliberately stops at a JSON manifest.  This module is
the boundary between that manifest and the synthetic/injection pipeline: it
preserves the cadence and observation metadata from MAST while allowing the
array reader to be supplied by a downloader, a FITS reader, or a test
fixture.  No timestamp is silently relabeled as BJD_TDB.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from synthetic.parents import RealExposureParent, RealObservationParent

from .models import ManifestRecord

ArrayLoader = Callable[[ManifestRecord], Mapping[str, Any]]
TimeConverter = Callable[[ManifestRecord], tuple[float, float]]


class ManifestParentLoader:
    """Convert manifest records into a source-centered real-parent bundle.

    ``array_loader`` is intentionally dependency-injected.  A production
    caller can resolve MAST download URIs to local FITS files, while unit tests
    and streaming jobs can provide memory-mapped arrays without importing
    Astropy.  The loader accepts optional values documented in
    :meth:`_exposure_values`; unknown values are retained in provenance rather
    than discarded.
    """

    def __init__(
        self,
        array_loader: ArrayLoader,
        *,
        time_converter: TimeConverter | None = None,
        time_converter_label: str | None = None,
    ) -> None:
        if not callable(array_loader):
            raise TypeError("array_loader must be callable")
        self.array_loader = array_loader
        self.time_converter = time_converter
        self.time_converter_label = time_converter_label or ("none" if time_converter is None else "explicit_converter")

    def load(
        self,
        records: Sequence[ManifestRecord],
        *,
        target_id: str,
        source_x: float,
        source_y: float,
        observation_id: str | None = None,
    ) -> RealObservationParent:
        if not records:
            raise ValueError("at least one manifest record is required")
        if not str(target_id).strip():
            raise ValueError("target_id must be non-empty")

        exposures: list[RealExposureParent] = []
        for record in sorted(records, key=self._sort_key):
            arrays = dict(self.array_loader(record))
            start, end = self._bjd_window(record)
            values = self._exposure_values(record, arrays, start, end)
            exposures.append(RealExposureParent(**values))

        parent_id = observation_id or target_id
        return RealObservationParent(
            observation_id=parent_id,
            target_id=target_id,
            source_x=float(source_x),
            source_y=float(source_y),
            exposures=tuple(exposures),
            provenance={
                "source": "MAST",
                "manifest_records": len(records),
                "observation_ids": tuple(record.observation_id for record in records),
                "product_ids": tuple(record.product_id for record in records),
                "time_system": "BJD_TDB",
                "time_conversion": self.time_converter_label,
                "cadence_policy": "exact_manifest_windows",
            },
        )

    def _bjd_window(self, record: ManifestRecord) -> tuple[float, float]:
        declared = (record.time_system or "").strip().upper().replace("-", "_")
        if declared in {"BJD_TDB", "BJDTDB", "BJD TDB"}:
            start = _finite_float(record.observation_start, "observation_start")
            end = _finite_float(record.observation_end, "observation_end")
            if end <= start:
                raise ValueError(f"manifest record {record.observation_id} has an invalid time window")
            return start, end
        if self.time_converter is None:
            raise ValueError(
                f"manifest record {record.observation_id} uses {record.time_system!r}; "
                "provide an explicit time_converter to produce BJD_TDB"
            )
        converted = self.time_converter(record)
        if not isinstance(converted, Sequence) or len(converted) != 2:
            raise ValueError("time_converter must return (start_bjd_tdb, end_bjd_tdb)")
        start = _finite_float(converted[0], "converted start")
        end = _finite_float(converted[1], "converted end")
        if end <= start:
            raise ValueError(f"time_converter returned an invalid window for {record.observation_id}")
        return start, end

    @staticmethod
    def _sort_key(record: ManifestRecord) -> float:
        try:
            return float(record.observation_start)
        except (TypeError, ValueError):
            return float("inf")

    @staticmethod
    def _exposure_values(
        record: ManifestRecord,
        arrays: dict[str, Any],
        start: float,
        end: float,
    ) -> dict[str, Any]:
        footprint = record.spatial_footprint or {}
        coverage = record.coverage or {}
        pointing = dict(record.pointing or {})
        if isinstance(arrays.get("pointing"), Mapping):
            pointing.update(arrays["pointing"])

        instrument = str(arrays.get("instrument") or record.instrument or "UNKNOWN")
        detector = str(
            arrays.get("detector")
            or footprint.get("detector")
            or _detector_from_instrument(instrument)
            or "UNKNOWN"
        )
        filter_name = str(
            arrays.get("filter_name")
            or record.wavelength.passband
            or "UNKNOWN"
        )
        exposure_id = str(arrays.get("exposure_id") or record.product_id or record.observation_id)
        visit_id = str(
            arrays.get("visit_id")
            or coverage.get("visit_id")
            or record.observation_id
        )
        duration = record.exposure_duration_seconds
        if duration is None:
            duration = (end - start) * 86400.0

        pixel_scale = _pair(
            arrays.get("pixel_scale_arcsec")
            or footprint.get("s_pixel_scale")
            or coverage.get("pixel_scale_arcsec"),
            default=(0.04, 0.04),
        )
        angular_size = _pair(
            arrays.get("angular_size_arcsec")
            or footprint.get("s_fov")
            or coverage.get("angular_size_arcsec"),
            default=None,
        )
        observer_position = _vector3(arrays.get("observer_position") or record.observer_position)
        observer_velocity = _vector3(arrays.get("observer_velocity") or record.observer_velocity)

        known_array_keys = {
            "science", "uncertainty", "dq", "wcs", "pointing", "instrument",
            "detector", "filter_name", "exposure_id", "visit_id",
            "pixel_scale_arcsec", "observer_position", "observer_velocity",
            "gain_electrons_per_adu", "read_noise_electrons", "saturation_electrons",
            "focus", "jitter", "read_times_seconds", "previous_exposure_fluence",
            "previous_exposure_time_bjd_tdb",
        }
        provenance = {
            "source": "MAST",
            "observation_id": record.observation_id,
            "product_id": record.product_id,
            "product_uri": record.product_uri,
            "download_uri": record.download_uri,
            "calibration_level": record.calibration_level,
            "product_type": record.product_type,
            "time_system": "BJD_TDB",
            "wavelength": record.wavelength.to_dict() if hasattr(record.wavelength, "to_dict") else {
                "minimum_nm": record.wavelength.minimum_nm,
                "maximum_nm": record.wavelength.maximum_nm,
                "passband": record.wavelength.passband,
            },
            "unconsumed_array_metadata": {
                key: value for key, value in arrays.items() if key not in known_array_keys
            },
        }
        return {
            "exposure_id": exposure_id,
            "visit_id": visit_id,
            "instrument": instrument,
            "detector": detector,
            "filter_name": filter_name,
            "t_start_bjd_tdb": start,
            "t_end_bjd_tdb": end,
            "exposure_duration_seconds": float(duration),
            "science": arrays.get("science"),
            "uncertainty": arrays.get("uncertainty"),
            "dq": arrays.get("dq"),
            "gain_electrons_per_adu": float(arrays.get("gain_electrons_per_adu", 1.0)),
            "read_noise_electrons": float(arrays.get("read_noise_electrons", 0.0)),
            "saturation_electrons": float(arrays.get("saturation_electrons", 65535.0)),
            "pixel_scale_arcsec": pixel_scale,
            "angular_size_arcsec": angular_size,
            "wcs": arrays.get("wcs", record.wcs_uri),
            "pointing": pointing,
            "observer_position": observer_position,
            "observer_velocity": observer_velocity,
            "focus": _optional_float(arrays.get("focus")),
            "jitter": arrays.get("jitter"),
            "read_times_seconds": tuple(arrays.get("read_times_seconds", ())),
            "previous_exposure_fluence": arrays.get("previous_exposure_fluence"),
            "previous_exposure_time_bjd_tdb": _optional_float(
                arrays.get("previous_exposure_time_bjd_tdb")
            ),
            "provenance": provenance,
        }


class FitsManifestParentLoader(ManifestParentLoader):
    """Optional FITS-backed loader for downloaded MAST products.

    Astropy is imported only when a record is read.  This keeps the REST and
    contract tests lightweight while providing a direct path from a manifest
    product to the parent bundle in an environment with FITS support.
    """

    def __init__(
        self,
        paths: Mapping[str, str | Path],
        *,
        time_converter: TimeConverter | None = None,
        time_converter_label: str | None = None,
        target_shape: tuple[int, int] | None = None,
    ) -> None:
        self.paths = {str(key): Path(value) for key, value in paths.items()}
        self.target_shape = target_shape
        super().__init__(
            self._load_fits,
            time_converter=time_converter,
            time_converter_label=time_converter_label,
        )

    def _load_fits(self, record: ManifestRecord) -> Mapping[str, Any]:
        path = self.paths.get(str(record.product_id)) or self.paths.get(record.observation_id)
        if path is None:
            raise KeyError(f"no local FITS path registered for {record.product_id or record.observation_id}")
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise RuntimeError("FitsManifestParentLoader requires astropy") from exc

        with fits.open(path, memmap=True) as hdul:
            science = _first_hdu(hdul, "SCI", 0)
            uncertainty = _first_hdu(hdul, "ERR", None)
            dq = _first_hdu(hdul, "DQ", None)
            header = hdul[0].header
            return {
                "science": self._patch(None if science is None else np.asarray(science.data)),
                "uncertainty": self._patch(None if uncertainty is None else np.asarray(uncertainty.data)),
                "dq": self._patch(None if dq is None else np.asarray(dq.data)),
                "wcs": record.wcs_uri,
                "pointing": {
                    key: header[key]
                    for key in ("PA_APER", "RA_APER", "DEC_APER")
                    if key in header
                },
                "instrument": header.get("INSTRUME", record.instrument),
                "detector": header.get("DETECTOR"),
                "filter_name": header.get("FILTER", record.wavelength.passband),
                "gain_electrons_per_adu": header.get("CCDGAIN", 1.0),
                "read_noise_electrons": header.get("READNSEA", 0.0),
                "saturation_electrons": header.get("SATURATE", 65535.0),
            }

    def _patch(self, value: np.ndarray | None) -> np.ndarray | None:
        if value is None or self.target_shape is None:
            return value
        if value.ndim != 2:
            raise ValueError("FITS parent patching currently requires two-dimensional image arrays")
        target_height, target_width = self.target_shape
        output = np.zeros((target_height, target_width), dtype=value.dtype)
        source_y = max((value.shape[0] - target_height) // 2, 0)
        source_x = max((value.shape[1] - target_width) // 2, 0)
        read_height = min(value.shape[0], target_height)
        read_width = min(value.shape[1], target_width)
        dest_y = max((target_height - value.shape[0]) // 2, 0)
        dest_x = max((target_width - value.shape[1]) // 2, 0)
        output[dest_y:dest_y + read_height, dest_x:dest_x + read_width] = value[
            source_y:source_y + read_height, source_x:source_x + read_width
        ]
        return output


def _first_hdu(hdul: Any, name: str, fallback_index: int | None) -> Any:
    for hdu in hdul:
        if str(getattr(hdu, "name", "")).upper() == name:
            return hdu
    if fallback_index is not None and len(hdul) > fallback_index:
        return hdul[fallback_index]
    return None


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_float(value: Any) -> float | None:
    return None if value is None else _finite_float(value, "optional float")


def _pair(
    value: Any, *, default: tuple[float, float] | None
) -> tuple[float, float] | None:
    if value is None:
        return default
    if isinstance(value, Mapping):
        fallback_x = default[0] if default is not None else None
        fallback_y = default[1] if default is not None else None
        value = (
            value.get("x", value.get("ra", fallback_x)),
            value.get("y", value.get("dec", fallback_y)),
        )
    if isinstance(value, (str, bytes)):
        return default
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return default
    if len(values) == 1:
        values = (values[0], values[0])
    if len(values) != 2 or any(not np.isfinite(item) or item <= 0.0 for item in values):
        return default
    return values


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = (value.get("x"), value.get("y"), value.get("z"))
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(values) != 3 or any(not np.isfinite(item) for item in values):
        return None
    return values


def _detector_from_instrument(instrument: str) -> str | None:
    if "/" in instrument:
        return instrument.rsplit("/", 1)[1]
    return None
