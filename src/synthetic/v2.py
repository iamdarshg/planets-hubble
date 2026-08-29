"""Opt-in Hubble Synthetic v2 orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .injection import ParentInjectionResult, RealParentInjector
from .parents import RealObservationParent
from .population import ObservationProgram, PopulationDraw, PopulationSampler


@dataclass(frozen=True)
class HubbleSyntheticV2Result:
    population: PopulationDraw
    injection: ParentInjectionResult


class HubbleSyntheticV2:
    """Combine coupled population draws with real-parent injection.

    This façade intentionally requires a loaded parent observation.  It makes
    it difficult for callers to mistake idealized R0 samples for R4/R5 data.
    """

    def __init__(self, *, seed: int = 0, injector: RealParentInjector | None = None) -> None:
        self.population_sampler = PopulationSampler(seed=seed)
        self.injector = injector or RealParentInjector()

    def generate(self, parent: RealObservationParent, *, sample_index: int = 0) -> HubbleSyntheticV2Result:
        draw = self.population_sampler.draw(sample_index=sample_index)
        # The real parent owns the observing mode.  Never let a synthetic
        # population draw silently change an HST UVIS parent into an IR or
        # Kepler observation.
        first_exposure = parent.exposures[0]
        draw = replace(
            draw,
            observation=ObservationProgram(
                instrument=first_exposure.instrument,
                detector=first_exposure.detector,
                filter_name=first_exposure.filter_name,
                exposure_seconds=first_exposure.exposure_seconds,
                visits=len(parent.visit_ids),
                local_steps=max(
                    sum(exposure.visit_id == visit_id for exposure in parent.exposures)
                    for visit_id in parent.visit_ids
                ),
            ),
        )
        if draw.planet is None:
            injection = self.injector.preserve_parent(parent, event_type=draw.event_type)
        else:
            first_midpoint = parent.exposures[0].t_mid_bjd_tdb
            injection = self.injector.inject_transit(
                parent,
                epoch_bjd_tdb=first_midpoint,
                period_days=draw.planet.period_days,
                depth=min(0.95, draw.planet.radius_ratio**2),
                duration_days=draw.planet.transit_duration_days,
            )
        return HubbleSyntheticV2Result(population=draw, injection=injection)
