"""Replay and block-bootstrap schedules from real observation parents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .parents import RealObservationParent


def _readonly(values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservationSchedule:
    """Exposure windows and provenance produced by a schedule sampler."""

    starts: np.ndarray
    mids: np.ndarray
    ends: np.ndarray
    exposure_ids: tuple[str, ...]
    visit_ids: tuple[str, ...]
    metadata: dict[str, Any]


class ObservationScheduleSampler:
    """Select whole real exposure windows without inventing cadence."""

    def __init__(self, parent: RealObservationParent) -> None:
        self.parent = parent

    def sample(self) -> ObservationSchedule:
        exposures = self.parent.exposures
        starts = _readonly([exposure.t_start_bjd_tdb for exposure in exposures])
        mids = _readonly([exposure.t_mid_bjd_tdb for exposure in exposures])
        ends = _readonly([exposure.t_end_bjd_tdb for exposure in exposures])
        # Preserve the actual gap after each exposure.  In particular, do not
        # derive this from a difference of differences: that loses the only
        # gap for a two-exposure parent and misaligns longer parents.
        gaps = starts[1:] - ends[:-1] if len(exposures) > 1 else np.empty(0)
        return ObservationSchedule(
            starts=starts,
            mids=mids,
            ends=ends,
            exposure_ids=tuple(exposure.exposure_id for exposure in exposures),
            visit_ids=tuple(exposure.visit_id for exposure in exposures),
            metadata={
                "mode": "real_parent_replay",
                "parent_observation_id": self.parent.observation_id,
                "gap_days": float(np.max(gaps)) if gaps.size else 0.0,
            },
        )

    def block_bootstrap(self, *, seed: int, visits: int) -> ObservationSchedule:
        if not isinstance(visits, int) or visits < 1:
            raise ValueError("visits must be a positive integer")
        visit_groups: dict[str, list[Any]] = {}
        for exposure in self.parent.exposures:
            visit_groups.setdefault(exposure.visit_id, []).append(exposure)
        visit_names = tuple(visit_groups)
        rng = np.random.default_rng(seed)
        selected_names = tuple(visit_names[index] for index in rng.integers(0, len(visit_names), size=visits))
        selected = [exposure for name in selected_names for exposure in visit_groups[name]]
        starts = _readonly([exposure.t_start_bjd_tdb for exposure in selected])
        mids = _readonly([exposure.t_mid_bjd_tdb for exposure in selected])
        ends = _readonly([exposure.t_end_bjd_tdb for exposure in selected])
        return ObservationSchedule(
            starts=starts,
            mids=mids,
            ends=ends,
            exposure_ids=tuple(exposure.exposure_id for exposure in selected),
            visit_ids=tuple(exposure.visit_id for exposure in selected),
            metadata={
                "mode": "whole_visit_block_bootstrap",
                "parent_observation_id": self.parent.observation_id,
                "seed": seed,
                "selected_visits": selected_names,
            },
        )
