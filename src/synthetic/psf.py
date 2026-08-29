"""Empirical-first PSF selection with a bounded physics fallback.

The library intentionally stores kernels supplied by a caller (for example a
small extract from an STScI empirical PSF library) instead of downloading or
committing the large archive.  Production data can populate the same API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PsfResult:
    kernel: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _PsfEntry:
    instrument: str
    detector: str
    filter_name: str
    x: float
    y: float
    focus: float
    kernel: np.ndarray


class EmpiricalPsfLibrary:
    """Small in-memory index for detector-position/focus PSF kernels."""

    def __init__(self) -> None:
        self._entries: list[_PsfEntry] = []

    def add(
        self,
        *,
        instrument: str,
        detector: str,
        filter_name: str,
        x: float,
        y: float,
        focus: float,
        kernel: np.ndarray,
    ) -> None:
        array = np.asarray(kernel, dtype=np.float32).copy()
        if array.ndim != 2 or min(array.shape) < 3 or any(size % 2 == 0 for size in array.shape):
            raise ValueError("PSF kernels must be odd-sized 2D arrays")
        if not np.all(np.isfinite(array)) or np.any(array < 0.0) or float(array.sum()) <= 0.0:
            raise ValueError("PSF kernels must be finite, non-negative, and non-empty")
        self._entries.append(
            _PsfEntry(instrument, detector, filter_name, float(x), float(y), float(focus), array / array.sum())
        )

    def lookup(
        self, *, instrument: str, detector: str, filter_name: str, x: float, y: float, focus: float
    ) -> tuple[np.ndarray, float] | None:
        candidates = [
            entry
            for entry in self._entries
            if entry.instrument == instrument
            and entry.detector == detector
            and entry.filter_name == filter_name
        ]
        if not candidates:
            return None
        distances = [
            math.hypot((entry.x - x) / 256.0, (entry.y - y) / 256.0)
            + abs(entry.focus - focus)
            for entry in candidates
        ]
        index = int(np.argmin(distances))
        selected = candidates[index]
        return selected.kernel.copy(), float(distances[index])


class PsfProvider:
    """Select an empirical kernel and otherwise render an optical approximation."""

    def __init__(self, empirical: EmpiricalPsfLibrary | None = None, *, kernel_size: int = 33) -> None:
        if kernel_size < 9 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 9")
        self.empirical = empirical
        self.kernel_size = kernel_size

    def render(
        self,
        *,
        instrument: str,
        detector: str,
        filter_name: str,
        x: float,
        y: float,
        wavelength_nm: float,
        focus: float = 0.0,
        jitter: tuple[float, float] = (0.0, 0.0),
    ) -> PsfResult:
        if wavelength_nm <= 0.0:
            raise ValueError("wavelength_nm must be positive")
        if self.empirical is not None:
            match = self.empirical.lookup(
                instrument=instrument, detector=detector, filter_name=filter_name, x=x, y=y, focus=focus
            )
            if match is not None:
                kernel, distance = match
                return PsfResult(
                    kernel=kernel,
                    metadata={
                        "tier": "empirical",
                        "instrument": instrument,
                        "detector": detector,
                        "filter_name": filter_name,
                        "matched_distance": distance,
                    },
                )
        return PsfResult(
            kernel=self._physics_kernel(wavelength_nm, focus, jitter, x, y),
            metadata={
                "tier": "physics_fallback",
                "instrument": instrument,
                "detector": detector,
                "filter_name": filter_name,
                "wavelength_nm": float(wavelength_nm),
                "focus": float(focus),
            },
        )

    def _physics_kernel(
        self,
        wavelength_nm: float,
        focus: float,
        jitter: tuple[float, float],
        x: float,
        y: float,
    ) -> np.ndarray:
        half = self.kernel_size // 2
        yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float64)
        # The scale is a compact diffraction approximation.  The ring term and
        # orthogonal spikes are intentional: a Gaussian-only PSF erases two
        # strong HST morphology cues needed by the model.
        scale = 0.65 + 0.0018 * wavelength_nm + 0.45 * abs(focus)
        x0, y0 = float(jitter[0]), float(jitter[1])
        radius = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
        core = np.sinc(radius / max(scale, 0.2)) ** 2
        ring = 0.16 * np.sinc(radius / max(scale * 2.2, 0.3)) ** 2
        spike_strength = 0.025 * (1.0 + abs(focus))
        spikes = spike_strength * (np.exp(-0.5 * (xx / 0.7) ** 2) + np.exp(-0.5 * (yy / 0.7) ** 2))
        asymmetry = 1.0 + 0.04 * math.sin((x + y) / 400.0) * xx / max(half, 1)
        kernel = np.clip((core + ring + spikes) * asymmetry, 0.0, None)
        kernel /= kernel.sum()
        return kernel.astype(np.float32)
