"""Transparent candidate extraction from model heatmaps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateRecord:
    x: int
    y: int
    candidate_probability: float
    event_probability: float
    transit_time_bjd_tdb: float
    time_system: str = "BJD_TDB"


class HeatmapCandidateClassifier:
    """Small non-neural baseline that ranks spatial-temporal heatmap peaks."""

    def __init__(self, *, threshold: float = 0.5, max_candidates: int = 32, suppression_radius: int = 3) -> None:
        if not 0.0 <= threshold <= 1.0 or max_candidates < 1 or suppression_radius < 0:
            raise ValueError("invalid candidate-classifier settings")
        self.threshold = float(threshold)
        self.max_candidates = int(max_candidates)
        self.suppression_radius = int(suppression_radius)

    def predict(
        self,
        candidate_heatmap: np.ndarray,
        *,
        event_probability: np.ndarray,
        timestamps_bjd_tdb: np.ndarray,
    ) -> tuple[CandidateRecord, ...]:
        heatmap = np.asarray(candidate_heatmap, dtype=np.float32)
        events = np.asarray(event_probability, dtype=np.float32)
        times = np.asarray(timestamps_bjd_tdb, dtype=np.float64)
        if heatmap.ndim != 4:
            raise ValueError("candidate_heatmap must have shape [visits, steps, height, width]")
        visits, steps, height, width = heatmap.shape
        if events.shape != (visits, steps) or times.shape != (visits, steps):
            raise ValueError("event_probability and timestamps must have shape [visits, steps]")
        if not np.isfinite(heatmap).all() or not np.isfinite(events).all() or not np.isfinite(times).all():
            raise ValueError("classifier inputs must be finite")
        score = np.clip(heatmap, 0.0, 1.0) * np.clip(events, 0.0, 1.0)[:, :, None, None]
        candidates = np.argwhere(score >= self.threshold)
        ordered = sorted(candidates.tolist(), key=lambda index: float(score[tuple(index)]), reverse=True)
        records: list[CandidateRecord] = []
        for visit, step, y, x in ordered:
            if any(
                (x - record.x) ** 2 + (y - record.y) ** 2
                <= self.suppression_radius**2
                for record in records
            ):
                continue
            time_index = np.unravel_index(int(np.argmax(score[:, :, y, x])), (visits, steps))
            records.append(
                CandidateRecord(
                    x=int(x),
                    y=int(y),
                    candidate_probability=float(score[visit, step, y, x]),
                    event_probability=float(events[time_index]),
                    transit_time_bjd_tdb=float(times[time_index]),
                )
            )
            if len(records) >= self.max_candidates:
                break
        return tuple(records)
