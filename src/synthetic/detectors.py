"""Bounded detector-level approximations for WFC3 UVIS and IR observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DetectorOutput:
    signal: np.ndarray
    uncertainty: np.ndarray
    dq: np.ndarray
    metadata: dict[str, object]
    read_stack: np.ndarray | None = None

    SATURATED_BIT = 1
    COSMIC_RAY_BIT = 2
    PERSISTENCE_BIT = 4


@dataclass(frozen=True)
class _HistoryEntry:
    fluence: np.ndarray
    time_bjd_tdb: float
    saturated: bool


class DetectorHistory:
    """Bounded detector memory used by IR persistence and sequence rendering."""

    def __init__(self, *, max_entries: int = 8) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._entries: deque[_HistoryEntry] = deque(maxlen=max_entries)

    @classmethod
    def from_parent(cls, exposure: object, *, max_entries: int = 8) -> "DetectorHistory":
        """Seed bounded detector memory from one loaded parent exposure."""
        history = cls(max_entries=max_entries)
        fluence = exposure.previous_exposure_fluence
        if fluence is None:
            return history
        if exposure.previous_exposure_time_bjd_tdb is None:
            raise ValueError("previous_exposure_time_bjd_tdb is required with previous fluence")
        history.record(
            fluence,
            time_bjd_tdb=exposure.previous_exposure_time_bjd_tdb,
        )
        return history

    def record(self, fluence: np.ndarray, *, time_bjd_tdb: float, saturated: bool = False) -> None:
        array = np.asarray(fluence, dtype=np.float32)
        if array.ndim != 2 or not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError("fluence must be a finite, non-negative 2D array")
        timestamp = float(time_bjd_tdb)
        if not np.isfinite(timestamp):
            raise ValueError("time_bjd_tdb must be finite")
        self._entries.append(_HistoryEntry(array.copy(), timestamp, bool(saturated)))

    def persistence(
        self,
        *,
        shape: tuple[int, int],
        now_bjd_tdb: float,
        amplitude: float,
        decay_days: float = 0.1,
    ) -> np.ndarray:
        if amplitude < 0.0 or decay_days <= 0.0:
            raise ValueError("amplitude must be non-negative and decay_days positive")
        result = np.zeros(shape, dtype=np.float32)
        for entry in reversed(self._entries):
            if entry.fluence.shape != shape or entry.time_bjd_tdb > now_bjd_tdb:
                continue
            elapsed = now_bjd_tdb - entry.time_bjd_tdb
            decay = math.exp(-elapsed / decay_days)
            normalized_fluence = np.minimum(entry.fluence / 100_000.0, 1.0)
            saturation_factor = 1.25 if entry.saturated else 1.0
            result += amplitude * saturation_factor * normalized_fluence * decay
        return result


class WFC3UVISSimulator:
    """CCD-like UVIS path with spatial CTE trailing and saturation flags."""

    def __init__(
        self,
        *,
        dark_electrons: float = 0.02,
        read_noise_electrons: float = 3.0,
        saturation_electrons: float = 65_000.0,
        cte_trailing: float = 0.0,
        cosmic_ray_rate: float = 0.0,
    ) -> None:
        if (
            not np.isfinite(dark_electrons)
            or not np.isfinite(read_noise_electrons)
            or not np.isfinite(saturation_electrons)
            or dark_electrons < 0.0
            or read_noise_electrons < 0.0
            or saturation_electrons <= 0.0
            or not 0.0 <= cte_trailing < 1.0
            or not 0.0 <= cosmic_ray_rate <= 1.0
        ):
            raise ValueError("cte_trailing and cosmic_ray_rate must be in [0, 1)")
        self.dark_electrons = float(dark_electrons)
        self.read_noise_electrons = float(read_noise_electrons)
        self.saturation_electrons = float(saturation_electrons)
        self.cte_trailing = float(cte_trailing)
        self.cosmic_ray_rate = float(cosmic_ray_rate)

    def observe(self, expected_electrons: np.ndarray, *, rng: np.random.Generator) -> DetectorOutput:
        expected = np.asarray(expected_electrons, dtype=np.float32)
        if expected.ndim != 2 or np.any(expected < 0.0) or not np.all(np.isfinite(expected)):
            raise ValueError("expected_electrons must be a finite, non-negative 2D array")
        raw = rng.poisson(expected + self.dark_electrons).astype(np.float32)
        dq = np.zeros(raw.shape, dtype=np.uint16)
        cosmic = rng.random(raw.shape) < self.cosmic_ray_rate
        if np.any(cosmic):
            raw[cosmic] += 20_000.0
            dq[cosmic] |= DetectorOutput.COSMIC_RAY_BIT

        if self.cte_trailing > 0.0:
            trailing = np.zeros_like(raw)
            for lag in range(1, min(12, raw.shape[0] - 1) + 1):
                shifted = np.zeros_like(raw)
                shifted[lag:] = raw[:-lag]
                trailing += self.cte_trailing * (1.0 - self.cte_trailing) ** (lag - 1) * shifted
            raw += trailing

        saturated = raw >= self.saturation_electrons
        dq[saturated] |= DetectorOutput.SATURATED_BIT
        signal = np.minimum(raw, self.saturation_electrons)
        uncertainty = np.sqrt(np.maximum(signal, 1.0) + self.read_noise_electrons**2).astype(np.float32)
        return DetectorOutput(
            signal=signal.astype(np.float32),
            uncertainty=uncertainty,
            dq=dq,
            metadata={
                "instrument": "WFC3",
                "detector": "UVIS",
                "tier": "R3",
                "approximation": "spatial_cte_trailing",
                "cte_spatial": True,
            },
        )


class WFC3IRSimulator:
    """IR MULTIACCUM simulator with ramp fitting and detector memory."""

    def __init__(
        self,
        *,
        read_times_seconds: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0),
        read_noise_electrons: float = 15.0,
        saturation_electrons: float = 80_000.0,
        nonlinearity_amplitude: float = 0.01,
        persistence_amplitude: float = 0.0,
        cosmic_ray_rate: float = 0.0,
    ) -> None:
        if len(read_times_seconds) < 2 or any(
            b <= a for a, b in zip(read_times_seconds, read_times_seconds[1:])
        ):
            raise ValueError("read_times_seconds must contain at least two increasing times")
        if (
            not np.isfinite(saturation_electrons)
            or saturation_electrons <= 0.0
            or any(
                not np.isfinite(value) or value < 0.0
                for value in (
                    read_noise_electrons,
                    nonlinearity_amplitude,
                    persistence_amplitude,
                    cosmic_ray_rate,
                )
            )
        ):
            raise ValueError("detector parameters must be non-negative")
        if cosmic_ray_rate > 1.0:
            raise ValueError("cosmic_ray_rate must be in [0, 1]")
        self.read_times_seconds = tuple(float(value) for value in read_times_seconds)
        self.read_noise_electrons = float(read_noise_electrons)
        self.saturation_electrons = float(saturation_electrons)
        self.nonlinearity_amplitude = float(nonlinearity_amplitude)
        self.persistence_amplitude = float(persistence_amplitude)
        self.cosmic_ray_rate = float(cosmic_ray_rate)

    def observe(
        self,
        expected_rate: np.ndarray,
        *,
        rng: np.random.Generator,
        history: DetectorHistory | None = None,
        time_bjd_tdb: float = 0.0,
    ) -> DetectorOutput:
        rate = np.asarray(expected_rate, dtype=np.float32)
        if rate.ndim != 2 or np.any(rate < 0.0) or not np.all(np.isfinite(rate)):
            raise ValueError("expected_rate must be a finite, non-negative 2D array")
        shape = rate.shape
        persistence = (
            history.persistence(shape=shape, now_bjd_tdb=time_bjd_tdb, amplitude=self.persistence_amplitude)
            if history is not None and self.persistence_amplitude > 0.0
            else np.zeros(shape, dtype=np.float32)
        )
        stack = np.zeros((len(self.read_times_seconds), *shape), dtype=np.float32)
        previous_time = 0.0
        for index, current_time in enumerate(self.read_times_seconds):
            delta = current_time - previous_time
            if delta > 0.0:
                stack[index] = stack[index - 1] + rng.poisson(rate * delta).astype(np.float32)
            if index == 0:
                stack[index] = persistence
            else:
                stack[index] += persistence
            cosmic = rng.random(shape) < self.cosmic_ray_rate
            if np.any(cosmic):
                stack[index:][..., cosmic] += 25_000.0
            previous_time = current_time

        normalized = stack / max(self.saturation_electrons, 1.0)
        stack = stack + self.nonlinearity_amplitude * normalized**2 * self.saturation_electrons
        saturated = stack >= self.saturation_electrons
        dq = np.zeros(shape, dtype=np.uint16)
        if np.any(saturated):
            dq[np.any(saturated, axis=0)] |= DetectorOutput.SATURATED_BIT
        if np.any(persistence > 0.0):
            dq[persistence > 0.0] |= DetectorOutput.PERSISTENCE_BIT
        stack = np.minimum(stack, self.saturation_electrons)

        times = np.asarray(self.read_times_seconds, dtype=np.float64)
        design = np.column_stack((times, np.ones_like(times)))
        projection = np.linalg.pinv(design)
        signal = np.tensordot(projection[0], stack, axes=(0, 0)).astype(np.float32)
        uncertainty = np.sqrt(np.maximum(signal, 1.0) + self.read_noise_electrons**2).astype(np.float32)
        if history is not None:
            history.record(stack[-1], time_bjd_tdb=time_bjd_tdb, saturated=bool(np.any(saturated)))
        return DetectorOutput(
            signal=signal,
            uncertainty=uncertainty,
            dq=dq,
            read_stack=stack,
            metadata={
                "instrument": "WFC3",
                "detector": "IR",
                "tier": "R3",
                "approximation": "MULTIACCUM_ramp_fit",
                "readout_mode": "MULTIACCUM",
                "persistence_source": "detector_history"
                if history is not None and self.persistence_amplitude > 0.0
                else "none",
            },
        )
