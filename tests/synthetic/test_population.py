import math

import numpy as np

from synthetic.population import PopulationSampler


def test_population_draw_is_deterministic_and_coupled() -> None:
    first = PopulationSampler(seed=17).draw(sample_index=4, event_type="planet_transit")
    repeat = PopulationSampler(seed=17).draw(sample_index=4, event_type="planet_transit")
    other = PopulationSampler(seed=17).draw(sample_index=5, event_type="planet_transit")

    assert first == repeat
    assert first.observation.instrument in {"WFC3", "Kepler"}
    assert first.observation.visits >= 1
    assert first.planet.period_days != other.planet.period_days
    assert first.star.radius_solar > 0.0
    assert first.planet.semi_major_axis_au > first.star.radius_solar * 0.00465047


def test_kepler_relation_and_duration_are_derived_from_star_and_orbit() -> None:
    draw = PopulationSampler(seed=23).draw(sample_index=0, event_type="planet_transit")
    expected_axis = (
        draw.star.mass_solar * draw.planet.period_days**2 / 365.25**2
    ) ** (1.0 / 3.0)

    assert math.isclose(draw.planet.semi_major_axis_au, expected_axis, rel_tol=1e-12)
    assert 0.0 < draw.planet.transit_duration_days < draw.planet.period_days
    assert 0.0 < draw.planet.radius_ratio < 0.5
    assert -1.0 <= draw.planet.eccentricity < 1.0
    assert np.isfinite(draw.star.teff_kelvin)


def test_population_supports_hard_negative_event_classes() -> None:
    for event_type in ("eclipsing_binary", "stellar_variability", "null"):
        draw = PopulationSampler(seed=5).draw(sample_index=2, event_type=event_type)
        assert draw.event_type == event_type
        assert draw.planet is None or draw.planet.period_days > 0.0
        assert draw.observation.filter_name
