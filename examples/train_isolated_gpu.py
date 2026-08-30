"""Train the research model with one bounded CUDA process per sample view."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=16, help="counterfactual pairs")
    parser.add_argument("--start-seed", type=int, default=23)
    parser.add_argument(
        "--fixed-seed",
        action="store_true",
        help="repeat the same counterfactual pair for an overfit learnability probe",
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/training/synthetic.pt"))
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--event-only", action="store_true")
    parser.add_argument("--source-event-curriculum", action="store_true")
    args = parser.parse_args()
    if args.event_only and args.source_event_curriculum:
        raise ValueError("event-only and source-event-curriculum are mutually exclusive")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    worker = Path(__file__).with_name("isolated_gpu_step.py")
    for pair_index in range(args.steps):
        for view in (0, 1):
            seed = args.start_seed if args.fixed_seed else args.start_seed + pair_index
            completed = subprocess.run(
                [
                    sys.executable,
                    str(worker),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--seed",
                    str(seed),
                    "--view",
                    str(view),
                    "--learning-rate",
                    str(args.learning_rate),
                    *(["--event-only"] if args.event_only else []),
                    *(["--source-event-curriculum"] if args.source_event_curriculum else []),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                print(completed.stdout, end="")
                print(completed.stderr, end="", file=sys.stderr)
                return completed.returncode
            line = next(
                (item for item in reversed(completed.stdout.splitlines()) if item.strip().startswith("{")),
                "{}",
            )
            result = json.loads(line)
            result["pair"] = pair_index
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"steps": len(results), "checkpoint": str(args.checkpoint), "history": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
