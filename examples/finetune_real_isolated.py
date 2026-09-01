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
    parser.add_argument(
        "--single-exposure",
        action="store_true",
        help="use one exposure per update instead of the full parent time series",
    )
    parser.add_argument(
        "--sequence-summary",
        action="store_true",
        help="reduce the complete parent sequence to a cap-safe temporal summary",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="BF16-compatible SGD learning rate; use a smaller value only with FP32 weights",
    )
    parser.add_argument(
        "--source-event-curriculum",
        action="store_true",
        help="train global event and source-proposal heatmap objectives together",
    )
    parser.add_argument(
        "--skip-dense-heatmaps",
        action="store_true",
        help="pass --skip-dense-heatmaps to each worker so the fine-tune "
        "architecture matches decoder-off synthetic checkpoints.",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="use one shared positive/null optimizer step per iteration",
    )
    parser.add_argument(
        "--cudnn-off",
        action="store_true",
        help="disable cuDNN workspace allocation in each isolated worker",
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
    averager = Path(__file__).with_name("average_checkpoints.py")
    results: list[dict[str, object]] = []
    for step in range(args.steps):
        exposure_index = step % args.exposures if args.single_exposure else None
        if args.paired:
            current_checkpoint = args.checkpoint.read_bytes()
            positive_checkpoint = args.checkpoint.with_suffix(
                f".pair{step}.positive{args.checkpoint.suffix}"
            )
            negative_checkpoint = args.checkpoint.with_suffix(
                f".pair{step}.negative{args.checkpoint.suffix}"
            )
            positive_checkpoint.write_bytes(current_checkpoint)
            negative_checkpoint.write_bytes(current_checkpoint)
            del current_checkpoint
            views = (0, 1)
        else:
            positive_checkpoint = negative_checkpoint = args.checkpoint
            views = (0, 1)
        for view in views:
            worker_checkpoint = (
                positive_checkpoint if view == 0 else negative_checkpoint
            )
            worker_args = [
                sys.executable,
                str(worker),
                "--checkpoint", str(worker_checkpoint),
                "--seed", str(step),
                "--view", str(view),
                "--phase", "real",
                "--manifest", str(args.manifest),
                *( ["--exposure-index", str(exposure_index)] if exposure_index is not None else [] ),
                "--learning-rate", str(args.learning_rate),
            ]
            if args.source_event_curriculum:
                worker_args.append("--source-event-curriculum")
            if args.skip_dense_heatmaps:
                worker_args.append("--skip-dense-heatmaps")
            if args.sequence_summary:
                worker_args.append("--sequence-summary")
            if args.cudnn_off:
                worker_args.append("--cudnn-off")
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
        if args.paired:
            averaged = subprocess.run(
                [
                    sys.executable,
                    str(averager),
                    "--positive", str(positive_checkpoint),
                    "--negative", str(negative_checkpoint),
                    "--output", str(args.checkpoint),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if averaged.returncode != 0:
                print(averaged.stdout, end="")
                print(averaged.stderr, end="", file=sys.stderr)
                return averaged.returncode
            positive_checkpoint.unlink(missing_ok=True)
            negative_checkpoint.unlink(missing_ok=True)
    print(json.dumps({"steps": len(results), "history": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
