"""Coupled astrophysical and observation-program priors for synthetic data.

The sampler deliberately draws a physical system first and derives dependent
quantities such as semi-major axis and transit duration.  It is small enough
to use in bounded local tests, but its output is also a stable contract for a
future catalog/isochrone-backed sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

_RSUN_IN_AU = 0.00465047


@dataclass(frozen=True)
class StarParameters:
    mass_solar: float
    radius_solar: float
    teff_kelvin: float
    metallicity_dex: float
    distance_pc: float
    extinction_mag: float


@dataclass(frozen=True)
class PlanetParameters:
    kind: str
    radius_ratio: float
    period_days: float
    eccentricity: float
    omega_rad: float
    impact_parameter: float
    semi_major_axis_au: float
    transit_duration_days: float


@dataclass(frozen=True)
class ObservationProgram:
    instrument: str
    detector: str
    filter_name: str
    exposure_seconds: float
    visits: int
    local_steps: int


@dataclass(frozen=True)
class PopulationDraw:
    event_type: str
    star: StarParameters
    planet: PlanetParameters | None
    observation: ObservationProgram
    seed: int


class PopulationSampler:
    """Generate independent, reproducible physical systems from one seed."""

    _EVENT_TYPES = {"planet_transit", "eclipsing_binary", "stellar_variability", "null"}

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = int(seed)

    def draw(self, *, sample_index: int = 0, event_type: str | None = None) -> PopulationDraw:
        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if event_type is not None and event_type not in self._EVENT_TYPES:
            raise ValueError(f"unsupported event_type: {event_type}")
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(sample_index)]))

        mass = float(np.exp(rng.uniform(np.log(0.55), np.log(1.45))))
        radius = float(mass ** 0.82 * np.exp(rng.normal(0.0, 0.04)))
        teff = float(5772.0 * mass ** 0.48 * np.exp(rng.normal(0.0, 0.015)))
        metallicity = float(np.clip(rng.normal(-0.05, 0.35), -1.5, 0.7))
        distance = float(np.exp(rng.uniform(np.log(20.0), np.log(2500.0))))
        extinction = float(np.clip(rng.exponential(0.08 * distance / 1000.0), 0.0, 1.5))
        star = StarParameters(mass, radius, teff, metallicity, distance, extinction)

        selected_event = event_type or str(
            rng.choice(("planet_transit", "eclipsing_binary", "stellar_variability", "null"), p=(0.45, 0.2, 0.2, 0.15))
        )
        planet = None
        if selected_event in {"planet_transit", "eclipsing_binary"}:
            period = float(np.exp(rng.uniform(np.log(0.5), np.log(120.0))))
            radius_ratio = (
                float(np.exp(rng.uniform(np.log(0.006), np.log(0.18))))
                if selected_event == "planet_transit"
                else float(rng.uniform(0.12, 0.8))
            )
            eccentricity = float(np.clip(rng.beta(1.2, 5.0) * 0.85, 0.0, 0.85))
            omega = float(rng.uniform(-math.pi, math.pi))
            impact = float(np.sqrt(rng.uniform(0.0, 0.95**2)))
            axis = float((mass * period**2 / 365.25**2) ** (1.0 / 3.0))
            duration = self._transit_duration_days(
                period, radius_ratio, radius, axis, eccentricity, omega, impact
            )
            planet = PlanetParameters(
                selected_event, radius_ratio, period, eccentricity, omega, impact, axis, duration
            )

        observation = self._observation_program(rng)
        return PopulationDraw(selected_event, star, planet, observation, self.seed + sample_index)

    @staticmethod
    def _transit_duration_days(
        period: float,
        radius_ratio: float,
        radius_solar: float,
        semi_major_axis_au: float,
        eccentricity: float,
        omega: float,
        impact: float,
    ) -> float:
        a_over_r = semi_major_axis_au / (_RSUN_IN_AU * radius_solar)
        cos_i = min(0.999999, max(1.0e-8, impact / max(a_over_r, 1.0e-8)))
        sin_i = math.sqrt(max(1.0e-12, 1.0 - cos_i * cos_i))
        chord = math.sqrt(max(1.0e-10, (1.0 + radius_ratio) ** 2 - impact**2))
        asin_argument = min(0.999999, max(1.0e-8, chord / (a_over_r * sin_i)))
        eccentric_factor = math.sqrt(1.0 - eccentricity**2) / max(
            1.0e-6, 1.0 + eccentricity * math.sin(omega)
        )
        return float(max(1.0e-6, period / math.pi * math.asin(asin_argument) * eccentric_factor))

    @staticmethod
    def _observation_program(rng: np.random.Generator) -> ObservationProgram:
        if bool(rng.integers(0, 5)):
            instrument = "WFC3"
            detector, filter_name = (
                ("UVIS", str(rng.choice(("F275W", "F336W", "F438W", "F606W", "F814W"))))
                if bool(rng.integers(0, 2))
                else ("IR", str(rng.choice(("F105W", "F125W", "F140W", "F160W"))))
            )
        else:
            instrument, detector, filter_name = "Kepler", "CCD", "Kepler-band"
        return ObservationProgram(
            instrument,
            detector,
            filter_name,
            float(np.exp(rng.uniform(np.log(30.0), np.log(1200.0)))),
            int(rng.integers(2, 13)),
            int(rng.integers(2, 17)),
        )
