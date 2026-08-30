"""Immutable contracts for real Hubble parent observations.

The V2 simulator treats a real observation as the source of truth for timing,
detector state, quality flags, and calibration provenance.  Arrays are copied
and made read-only at this boundary so a synthetic injection cannot mutate the
archival parent by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


def _readonly_array(value: Any, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value).copy()
    if array.ndim != 2 and name in {"science", "uncertainty", "dq"}:
        raise ValueError(f"{name} must be a two-dimensional image")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class RealExposureParent:
    """One real exposure and the metadata needed to replay it or inject it."""

    exposure_id: str
    visit_id: str
    instrument: str
    detector: str
    filter_name: str
    t_start_bjd_tdb: float
    t_end_bjd_tdb: float
    # MAST observation windows can cover a visit or scheduling interval rather
    # than the integration represented by one product.  Keep the physical
    # exposure duration explicit instead of deriving it from that window.
    exposure_duration_seconds: float | None = None
    science: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    dq: np.ndarray | None = None
    gain_electrons_per_adu: float = 1.0
    read_noise_electrons: float = 0.0
    saturation_electrons: float = 65535.0
    pixel_scale_arcsec: tuple[float, float] = (0.04, 0.04)
    angular_size_arcsec: tuple[float, float] | None = None
    wcs: Any = None
    pointing: Mapping[str, Any] = field(default_factory=dict)
    observer_position: tuple[float, float, float] | None = None
    observer_velocity: tuple[float, float, float] | None = None
    focus: float | None = None
    jitter: np.ndarray | None = None
    read_times_seconds: tuple[float, ...] = ()
    previous_exposure_fluence: np.ndarray | None = None
    previous_exposure_time_bjd_tdb: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("exposure_id", "visit_id", "instrument", "detector", "filter_name"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        start = float(self.t_start_bjd_tdb)
        end = float(self.t_end_bjd_tdb)
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            raise ValueError("t_end must be finite and greater than t_start")
        duration = self.exposure_duration_seconds
        if duration is not None:
            duration = float(duration)
            if not np.isfinite(duration) or duration <= 0.0:
                raise ValueError("exposure_duration_seconds must be finite and positive")
        if self.gain_electrons_per_adu <= 0.0:
            raise ValueError("gain_electrons_per_adu must be positive")
        if self.read_noise_electrons < 0.0 or self.saturation_electrons <= 0.0:
            raise ValueError("detector noise must be non-negative and saturation positive")
        if any(scale <= 0.0 for scale in self.pixel_scale_arcsec):
            raise ValueError("pixel_scale_arcsec values must be positive")
        angular_size = None
        if self.angular_size_arcsec is not None:
            angular_size = tuple(float(size) for size in self.angular_size_arcsec)
            if len(angular_size) != 2 or any(
                not np.isfinite(size) or size <= 0.0 for size in angular_size
            ):
                raise ValueError("angular_size_arcsec values must be finite and positive")

        science = _readonly_array(self.science, name="science")
        uncertainty = _readonly_array(self.uncertainty, name="uncertainty")
        dq = _readonly_array(self.dq, name="dq")
        shapes = {array.shape for array in (science, uncertainty, dq) if array is not None}
        if len(shapes) > 1:
            raise ValueError("science, uncertainty, and dq must have matching shapes")
        if uncertainty is not None and np.any(uncertainty < 0.0):
            raise ValueError("uncertainty must be non-negative")

        jitter = None if self.jitter is None else np.asarray(self.jitter, dtype=np.float32).copy()
        if jitter is not None:
            if jitter.ndim != 2 or jitter.shape[1] != 2:
                raise ValueError("jitter must have shape [time, 2]")
            jitter.setflags(write=False)
        previous = _readonly_array(self.previous_exposure_fluence, name="previous_exposure_fluence")
        if previous is not None and science is not None and previous.shape != science.shape:
            raise ValueError("previous_exposure_fluence must match science shape")
        if self.read_times_seconds and (
            any(time < 0.0 for time in self.read_times_seconds)
            or any(later <= earlier for earlier, later in zip(self.read_times_seconds, self.read_times_seconds[1:]))
        ):
            raise ValueError("read_times_seconds must be strictly increasing and non-negative")

        object.__setattr__(self, "t_start_bjd_tdb", start)
        object.__setattr__(self, "t_end_bjd_tdb", end)
        object.__setattr__(self, "exposure_duration_seconds", duration)
        object.__setattr__(self, "angular_size_arcsec", angular_size)
        object.__setattr__(self, "science", science)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "dq", dq)
        object.__setattr__(self, "jitter", jitter)
        object.__setattr__(self, "previous_exposure_fluence", previous)
        object.__setattr__(self, "pointing", dict(self.pointing))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def t_mid_bjd_tdb(self) -> float:
        return (self.t_start_bjd_tdb + self.t_end_bjd_tdb) / 2.0

    @property
    def exposure_seconds(self) -> float:
        if self.exposure_duration_seconds is not None:
            return self.exposure_duration_seconds
        return (self.t_end_bjd_tdb - self.t_start_bjd_tdb) * 86400.0


@dataclass(frozen=True)
class RealObservationParent:
    """A source-centered collection of real exposures from one observation."""

    observation_id: str
    target_id: str
    source_x: float
    source_y: float
    exposures: tuple[RealExposureParent, ...]
    object_tokens: np.ndarray | None = None
    object_mask: np.ndarray | None = None
    object_metadata: tuple[Mapping[str, Any], ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.observation_id).strip() or not str(self.target_id).strip():
            raise ValueError("observation_id and target_id must be non-empty")
        if not np.isfinite(self.source_x) or not np.isfinite(self.source_y):
            raise ValueError("source coordinates must be finite")
        exposures = tuple(self.exposures)
        if not exposures:
            raise ValueError("a real observation parent needs at least one exposure")
        if any(not isinstance(exposure, RealExposureParent) for exposure in exposures):
            raise TypeError("exposures must contain RealExposureParent values")
        object_tokens = None
        if self.object_tokens is not None:
            object_tokens = np.asarray(self.object_tokens, dtype=np.float32).copy()
            if object_tokens.ndim != 2:
                raise ValueError("object_tokens must have shape [objects, features]")
            object_tokens.setflags(write=False)
        object_mask = None
        if self.object_mask is not None:
            object_mask = np.asarray(self.object_mask, dtype=bool).copy()
            if object_mask.ndim != 1:
                raise ValueError("object_mask must have shape [objects]")
            if object_tokens is not None and object_mask.shape[0] != object_tokens.shape[0]:
                raise ValueError("object_mask and object_tokens must have matching object counts")
            object_mask.setflags(write=False)
        metadata = tuple(dict(item) for item in self.object_metadata)
        if object_tokens is not None and metadata and len(metadata) != object_tokens.shape[0]:
            raise ValueError("object_metadata and object_tokens must have matching object counts")
        object.__setattr__(self, "object_tokens", object_tokens)
        object.__setattr__(self, "object_mask", object_mask)
        object.__setattr__(self, "object_metadata", metadata)
        object.__setattr__(self, "exposures", exposures)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def visit_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(exposure.visit_id for exposure in self.exposures))
