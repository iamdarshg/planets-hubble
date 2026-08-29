"""Parent-conditioned astrophysical event injection."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .parents import RealObservationParent
from .psf import PsfProvider


@dataclass(frozen=True)
class InjectedExposure:
    exposure_id: str
    science: np.ndarray
    uncertainty: np.ndarray
    dq: np.ndarray
    relative_flux_drop: float


@dataclass(frozen=True)
class ParentInjectionResult:
    null: tuple[InjectedExposure, ...]
    injected: tuple[InjectedExposure, ...]
    transit_times_bjd_tdb: tuple[float, ...]
    metadata: dict[str, object]


class RealParentInjector:
    """Inject only astrophysical perturbations into an observed parent."""

    def __init__(self, psf_provider: PsfProvider | None = None) -> None:
        self.psf_provider = psf_provider or PsfProvider()

    def inject_transit(
        self,
        parent: RealObservationParent,
        *,
        epoch_bjd_tdb: float,
        period_days: float,
        depth: float,
        duration_days: float,
        source_flux_electrons: float | None = None,
        wavelength_nm: float | None = None,
    ) -> ParentInjectionResult:
        if period_days <= 0.0 or duration_days <= 0.0:
            raise ValueError("period_days and duration_days must be positive")
        if not 0.0 < depth < 1.0:
            raise ValueError("depth must be in (0, 1)")
        if not math.isfinite(epoch_bjd_tdb):
            raise ValueError("epoch_bjd_tdb must be finite")
        if source_flux_electrons is not None and source_flux_electrons <= 0.0:
            raise ValueError("source_flux_electrons must be positive")
        exposures = parent.exposures
        if any(exposure.science is None or exposure.uncertainty is None or exposure.dq is None for exposure in exposures):
            raise ValueError("all parent exposures need science, uncertainty, and dq arrays")
        start = min(exposure.t_start_bjd_tdb for exposure in exposures)
        end = max(exposure.t_end_bjd_tdb for exposure in exposures)
        first_k = math.ceil((start - epoch_bjd_tdb) / period_days - 0.5)
        last_k = math.floor((end - epoch_bjd_tdb) / period_days + 0.5)
        transit_times = tuple(
            epoch_bjd_tdb + index * period_days
            for index in range(first_k, last_k + 1)
            if start <= epoch_bjd_tdb + index * period_days <= end
        )

        null: list[InjectedExposure] = []
        injected: list[InjectedExposure] = []
        for exposure in exposures:
            science = np.asarray(exposure.science, dtype=np.float32)
            uncertainty = np.asarray(exposure.uncertainty, dtype=np.float32).copy()
            dq = np.asarray(exposure.dq, dtype=np.uint16).copy()
            drop = self._box_transit_drop(exposure.t_mid_bjd_tdb, transit_times, period_days, duration_days, depth)
            null.append(InjectedExposure(exposure.exposure_id, science.copy(), uncertainty.copy(), dq.copy(), 0.0))
            if drop == 0.0:
                injected_science = science.copy()
            else:
                filter_wavelength = wavelength_nm or self._filter_wavelength(exposure.filter_name)
                psf = self.psf_provider.render(
                    instrument=exposure.instrument,
                    detector=exposure.detector,
                    filter_name=exposure.filter_name,
                    x=parent.source_x,
                    y=parent.source_y,
                    wavelength_nm=filter_wavelength,
                    focus=exposure.focus or 0.0,
                    jitter=(0.0, 0.0),
                ).kernel
                source_flux = source_flux_electrons or float(max(1.0, science.max() * 5.0))
                loss = self._paste_at_source(
                    np.zeros_like(science), psf * float(source_flux * drop), parent.source_x, parent.source_y
                )
                injected_science = np.clip(science - loss, 0.0, None).astype(np.float32)
            injected.append(InjectedExposure(exposure.exposure_id, injected_science, uncertainty, dq, drop))
        return ParentInjectionResult(
            null=tuple(null),
            injected=tuple(injected),
            transit_times_bjd_tdb=transit_times,
            metadata={
                "mode": "real_parent_injection",
                "parent_observation_id": parent.observation_id,
                "target_id": parent.target_id,
                "cadence_source": parent.provenance.get("source", "unknown"),
            },
        )

    def preserve_parent(
        self, parent: RealObservationParent, *, event_type: str = "null"
    ) -> ParentInjectionResult:
        """Return a paired null example without changing the real parent."""
        null = tuple(
            InjectedExposure(
                exposure.exposure_id,
                np.asarray(exposure.science, dtype=np.float32).copy(),
                np.asarray(exposure.uncertainty, dtype=np.float32).copy(),
                np.asarray(exposure.dq, dtype=np.uint16).copy(),
                0.0,
            )
            for exposure in parent.exposures
        )
        return ParentInjectionResult(
            null=null,
            injected=tuple(
                InjectedExposure(item.exposure_id, item.science.copy(), item.uncertainty.copy(), item.dq.copy(), 0.0)
                for item in null
            ),
            transit_times_bjd_tdb=(),
            metadata={
                "mode": "real_parent_injection",
                "parent_observation_id": parent.observation_id,
                "target_id": parent.target_id,
                "event_type": event_type,
                "cadence_source": parent.provenance.get("source", "unknown"),
            },
        )

    @staticmethod
    def _box_transit_drop(
        time_bjd_tdb: float,
        transit_times: tuple[float, ...],
        period_days: float,
        duration_days: float,
        depth: float,
    ) -> float:
        if not transit_times:
            return 0.0
        phase_distance = min(
            abs(((time_bjd_tdb - transit) + period_days / 2.0) % period_days - period_days / 2.0)
            for transit in transit_times
        )
        return float(depth if phase_distance <= duration_days / 2.0 else 0.0)

    @staticmethod
    def _paste_at_source(canvas: np.ndarray, stamp: np.ndarray, x: float, y: float) -> np.ndarray:
        result = canvas.copy()
        height, width = result.shape
        half_y, half_x = stamp.shape[0] // 2, stamp.shape[1] // 2
        center_x, center_y = int(round(x)), int(round(y))
        y0, y1 = max(0, center_y - half_y), min(height, center_y + half_y + 1)
        x0, x1 = max(0, center_x - half_x), min(width, center_x + half_x + 1)
        sy0, sy1 = y0 - (center_y - half_y), y1 - (center_y - half_y)
        sx0, sx1 = x0 - (center_x - half_x), x1 - (center_x - half_x)
        result[y0:y1, x0:x1] += stamp[sy0:sy1, sx0:sx1]
        return result

    @staticmethod
    def _filter_wavelength(filter_name: str) -> float:
        return {
            "F275W": 270.0,
            "F336W": 335.0,
            "F438W": 432.0,
            "F606W": 590.0,
            "F814W": 800.0,
            "F105W": 1050.0,
            "F125W": 1250.0,
            "F140W": 1400.0,
            "F160W": 1540.0,
        }.get(filter_name, 600.0)
