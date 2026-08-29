import numpy as np

from synthetic import HubbleSyntheticV2
from synthetic.parents import RealExposureParent, RealObservationParent


def test_v2_uses_population_draw_and_parent_cadence() -> None:
    parent = RealObservationParent(
        observation_id="obs-v2",
        target_id="target-v2",
        source_x=2.0,
        source_y=2.0,
        exposures=(
            RealExposureParent(
                exposure_id="e1",
                visit_id="v1",
                instrument="WFC3",
                detector="IR",
                filter_name="F160W",
                t_start_bjd_tdb=10.0,
                t_end_bjd_tdb=10.01,
                science=np.full((5, 5), 100.0, dtype=np.float32),
                uncertainty=np.ones((5, 5), dtype=np.float32),
                dq=np.zeros((5, 5), dtype=np.uint16),
            ),
        ),
        provenance={"source": "MAST"},
    )
    result = HubbleSyntheticV2(seed=3).generate(parent, sample_index=0)

    assert result.population.observation.instrument in {"WFC3", "Kepler"}
    assert result.injection.metadata["mode"] == "real_parent_injection"
    assert result.injection.null[0].exposure_id == "e1"
    assert result.injection.injected[0].science.shape == (5, 5)
