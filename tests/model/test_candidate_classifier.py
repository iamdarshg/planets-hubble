import numpy as np

from model.candidate_classifier import HeatmapCandidateClassifier


def test_classifier_extracts_ranked_candidates_and_transit_time() -> None:
    candidate_heatmap = np.zeros((2, 3, 10, 12), dtype=np.float32)
    candidate_heatmap[1, 2, 4, 7] = 0.95
    event_probability = np.zeros((2, 3), dtype=np.float32)
    event_probability[1, 2] = 0.8
    timestamps = np.arange(6, dtype=np.float64).reshape(2, 3) + 2_460_000.0

    records = HeatmapCandidateClassifier(threshold=0.5, max_candidates=3).predict(
        candidate_heatmap, event_probability=event_probability, timestamps_bjd_tdb=timestamps
    )

    assert len(records) == 1
    assert records[0].x == 7
    assert records[0].y == 4
    assert records[0].transit_time_bjd_tdb == timestamps[1, 2]
    assert records[0].candidate_probability > 0.0
