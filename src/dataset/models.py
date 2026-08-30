"""Typed, JSON-friendly records used by the discovery prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SkyPosition:
    ra_deg: float
    dec_deg: float


@dataclass(frozen=True)
class WavelengthMetadata:
    minimum_nm: float | None = None
    maximum_nm: float | None = None
    passband: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestRecord:
    """One public product plus the observation metadata it came from."""

    observation_id: str
    product_id: str | None
    product_uri: str | None
    download_uri: str | None
    observation_start: Any = None
    observation_midpoint: Any = None
    observation_end: Any = None
    time_system: str | None = None
    exposure_duration_seconds: float | None = None
    wavelength: WavelengthMetadata = WavelengthMetadata()
    calibration_level: int | None = None
    product_type: str | None = None
    instrument: str | None = None
    wcs_uri: str | None = None
    spatial_footprint: dict[str, Any] | None = None
    observer_position: dict[str, Any] | None = None
    observer_velocity: dict[str, Any] | None = None
    pointing: dict[str, Any] | None = None
    coverage: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def start_time(self) -> Any:
        return self.observation_start

    @property
    def midpoint_time(self) -> Any:
        return self.observation_midpoint

    @property
    def end_time(self) -> Any:
        return self.observation_end

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiscoveryManifest:
    patch_id: str
    target_position: SkyPosition
    source_identifier: str | None
    radius_deg: float
    search_service: str
    records: tuple[ManifestRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
