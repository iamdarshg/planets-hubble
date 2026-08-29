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
    science: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    dq: np.ndarray | None = None
    gain_electrons_per_adu: float = 1.0
    read_noise_electrons: float = 0.0
    saturation_electrons: float = 65535.0
    pixel_scale_arcsec: tuple[float, float] = (0.04, 0.04)
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
        if self.gain_electrons_per_adu <= 0.0:
            raise ValueError("gain_electrons_per_adu must be positive")
        if self.read_noise_electrons < 0.0 or self.saturation_electrons <= 0.0:
            raise ValueError("detector noise must be non-negative and saturation positive")
        if any(scale <= 0.0 for scale in self.pixel_scale_arcsec):
            raise ValueError("pixel_scale_arcsec values must be positive")

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
        return (self.t_end_bjd_tdb - self.t_start_bjd_tdb) * 86400.0


@dataclass(frozen=True)
class RealObservationParent:
    """A source-centered collection of real exposures from one observation."""

    observation_id: str
    target_id: str
    source_x: float
    source_y: float
    exposures: tuple[RealExposureParent, ...]
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
        object.__setattr__(self, "exposures", exposures)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def visit_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(exposure.visit_id for exposure in self.exposures))
