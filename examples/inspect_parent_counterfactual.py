"""Measure whether a real-parent injection is visible in prepared pixels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic import HubbleSyntheticV2  # noqa: E402
from isolated_gpu_step import load_parent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--event-type",
        default=None,
        choices=("null", "planet_transit", "eclipsing_binary", "stellar_variability"),
        help="optional forced population event; defaults to the sampler draw",
    )
    args = parser.parse_args()
    parent = load_parent(args.manifest)
    result = HubbleSyntheticV2(seed=args.sample_index).generate(
        parent, sample_index=args.sample_index, event_type=args.event_type
    )
    rows = []
    for null, injected in zip(result.injection.null, result.injection.injected):
        difference = np.asarray(injected.science, dtype=np.float32) - np.asarray(
            null.science, dtype=np.float32
        )
        rows.append(
            {
                "exposure_id": injected.exposure_id,
                "relative_flux_drop": injected.relative_flux_drop,
                "difference_min": float(np.min(difference)),
                "difference_max": float(np.max(difference)),
                "difference_l1": float(np.sum(np.abs(difference))),
                "difference_nonzero_pixels": int(np.count_nonzero(difference)),
                "null_min": float(np.min(null.science)),
                "null_max": float(np.max(null.science)),
                "injected_min": float(np.min(injected.science)),
                "injected_max": float(np.max(injected.science)),
                "parent_max": float(np.max(null.science)),
            }
        )
    print(
        json.dumps(
            {
                "target": parent.target_id,
                "sample_index": args.sample_index,
                "event_type": result.population.event_type,
                "planet": result.population.planet is not None,
                "transit_times_bjd_tdb": result.injection.transit_times_bjd_tdb,
                "exposures": rows,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
