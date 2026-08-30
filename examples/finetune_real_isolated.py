"""Fine-tune the synthetic checkpoint on real-parent HST injections."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="checkpoint to update in place")
    parser.add_argument(
        "--input-checkpoint",
        type=Path,
        default=None,
        help="optional source checkpoint; copied to --checkpoint before the first worker",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--exposures", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--source-event-curriculum",
        action="store_true",
        help="train global event and source-proposal heatmap objectives together",
    )
    args = parser.parse_args()
    if args.steps < 1 or args.exposures < 1:
        raise ValueError("steps and exposures must be positive")
    if args.input_checkpoint is not None:
        if not args.input_checkpoint.is_file():
            raise FileNotFoundError(args.input_checkpoint)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_bytes(args.input_checkpoint.read_bytes())
    worker = Path(__file__).with_name("isolated_gpu_step.py")
    results: list[dict[str, object]] = []
    for step in range(args.steps):
        exposure_index = step % args.exposures
        for view in (0, 1):
            worker_args = [
                sys.executable,
                str(worker),
                "--checkpoint", str(args.checkpoint),
                "--seed", str(step),
                "--view", str(view),
                "--phase", "real",
                "--manifest", str(args.manifest),
                "--exposure-index", str(exposure_index),
                "--learning-rate", str(args.learning_rate),
            ]
            if args.source_event_curriculum:
                worker_args.append("--source-event-curriculum")
            completed = subprocess.run(
                worker_args,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                print(completed.stdout, end="")
                print(completed.stderr, end="", file=sys.stderr)
                return completed.returncode
            line = next(item for item in reversed(completed.stdout.splitlines()) if item.startswith("{"))
            result = json.loads(line)
            result["step"] = step
            result["exposure_index"] = exposure_index
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    print(json.dumps({"steps": len(results), "history": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
