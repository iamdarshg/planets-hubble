import numpy as np

from synthetic.detectors import DetectorHistory, WFC3IRSimulator, WFC3UVISSimulator


def test_persistence_depends_on_previous_fluence_and_decays_with_time() -> None:
    history = DetectorHistory(max_entries=2)
    fluence = np.zeros((4, 4), dtype=np.float32)
    fluence[1, 1] = 100_000.0
    history.record(fluence, time_bjd_tdb=10.0, saturated=True)

    recent = history.persistence(shape=(4, 4), now_bjd_tdb=10.01, amplitude=0.1)
    old = history.persistence(shape=(4, 4), now_bjd_tdb=11.0, amplitude=0.1)

    assert recent[1, 1] > old[1, 1] > 0.0
    assert recent[0, 0] == 0.0


def test_uvis_has_spatial_cte_trailing_and_saturation_flags() -> None:
    expected = np.zeros((8, 8), dtype=np.float32)
    expected[2, 3] = 20_000.0
    output = WFC3UVISSimulator(saturation_electrons=10_000.0, cte_trailing=0.08).observe(
        expected, rng=np.random.default_rng(4)
    )

    assert output.signal.shape == expected.shape
    assert output.signal[3, 3] > 0.0
    assert output.dq[2, 3] & output.SATURATED_BIT
    assert output.metadata["detector"] == "UVIS"


def test_ir_generates_nondestructive_ramp_and_fits_slope() -> None:
    history = DetectorHistory()
    output = WFC3IRSimulator(
        read_times_seconds=(0.0, 2.0, 4.0, 6.0),
        persistence_amplitude=0.02,
    ).observe(
        expected_rate=np.full((4, 4), 100.0, dtype=np.float32),
        rng=np.random.default_rng(8),
        history=history,
        time_bjd_tdb=20.0,
    )

    assert output.read_stack.shape == (4, 4, 4)
    assert output.signal.shape == (4, 4)
    assert np.isfinite(output.signal).all()
    assert output.metadata["readout_mode"] == "MULTIACCUM"
    assert output.metadata["detector"] == "IR"
