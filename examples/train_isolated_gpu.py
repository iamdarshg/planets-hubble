"""Train the research model with one bounded CUDA process per sample view."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


VIEW_LABELS = ("null", "injected")


def counterfactual_checkpoint_paths(
    checkpoint: Path, pair_index: int
) -> tuple[Path, Path]:
    """Return isolated positive and null checkpoints for one synthetic pair."""

    return (
        checkpoint.with_suffix(f".pair{pair_index}.positive{checkpoint.suffix}"),
        checkpoint.with_suffix(f".pair{pair_index}.negative{checkpoint.suffix}"),
    )


def label_for_view(view: int) -> str:
    if view not in (0, 1):
        raise ValueError("view must be 0 (null) or 1 (injected)")
    return VIEW_LABELS[view]


def checkpoint_for_view(positive: Path, negative: Path, view: int) -> Path:
    label_for_view(view)
    return negative if view == 0 else positive


def synthetic_worker_shape_args(
    visits: int, local_steps: int, *, skip_dense_heatmaps: bool = False
) -> list[str]:
    """Build validated shape flags shared by both counterfactual workers."""

    if visits < 1 or local_steps < 1:
        raise ValueError("visits and local_steps must be positive")
    return [
        "--visits", str(visits),
        "--local-steps", str(local_steps),
        *(["--skip-dense-heatmaps"] if skip_dense_heatmaps else []),
    ]


def build_worker_command(
    *,
    worker: Path,
    checkpoint: Path,
    seed: int,
    view: int,
    learning_rate: float,
    visits: int,
    local_steps: int,
    device: str,
    skip_dense_heatmaps: bool = False,
    event_only: bool = False,
    source_event_curriculum: bool = False,
) -> list[str]:
    """Build one isolated worker invocation with the full run contract."""

    label_for_view(view)
    return [
        sys.executable,
        str(worker),
        "--checkpoint", str(checkpoint),
        "--seed", str(seed),
        "--view", str(view),
        "--learning-rate", str(learning_rate),
        "--device", device,
        *synthetic_worker_shape_args(
            visits, local_steps, skip_dense_heatmaps=skip_dense_heatmaps
        ),
        *(["--event-only"] if event_only else []),
        *(["--source-event-curriculum"] if source_event_curriculum else []),
    ]


def temporary_checkpoint_path(checkpoint: Path) -> Path:
    return checkpoint.with_suffix(checkpoint.suffix + ".tmp")


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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--visits", type=int, default=1)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument(
        "--skip-dense-heatmaps",
        action="store_true",
        help="disable dense decoder activations for longer temporal sequences",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="train positive and null views from the same checkpoint, then average updates",
    )
    args = parser.parse_args()
    if args.event_only and args.source_event_curriculum:
        raise ValueError("event-only and source-event-curriculum are mutually exclusive")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    shape_args = synthetic_worker_shape_args(
        args.visits, args.local_steps, skip_dense_heatmaps=args.skip_dense_heatmaps
    )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    worker = Path(__file__).with_name("isolated_gpu_step.py")
    averager = Path(__file__).with_name("average_checkpoints.py")
    for pair_index in range(args.steps):
        positive_checkpoint: Path | None = None
        negative_checkpoint: Path | None = None
        try:
            if args.paired:
                if not args.checkpoint.is_file():
                    raise FileNotFoundError(
                        f"paired training requires an existing checkpoint: {args.checkpoint}"
                    )
                positive_checkpoint, negative_checkpoint = counterfactual_checkpoint_paths(
                    args.checkpoint, pair_index
                )
                snapshot = args.checkpoint.read_bytes()
                positive_checkpoint.write_bytes(snapshot)
                negative_checkpoint.write_bytes(snapshot)
                del snapshot
            else:
                positive_checkpoint = negative_checkpoint = args.checkpoint
            worker_checkpoints = {
                0: negative_checkpoint,
                1: positive_checkpoint,
            }
            for view in (0, 1):
                seed = args.start_seed if args.fixed_seed else args.start_seed + pair_index
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(worker),
                        "--checkpoint",
                        str(worker_checkpoints[view]),
                        "--seed",
                        str(seed),
                        "--view",
                        str(view),
                        "--learning-rate",
                        str(args.learning_rate),
                        "--device",
                        args.device,
                        *shape_args,
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
                result["view"] = view
                result["label"] = label_for_view(view)
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
        finally:
            if args.paired and positive_checkpoint is not None:
                positive_checkpoint.unlink(missing_ok=True)
            if args.paired and negative_checkpoint is not None:
                negative_checkpoint.unlink(missing_ok=True)
            temporary_checkpoint_path(args.checkpoint).unlink(missing_ok=True)
    print(json.dumps({"steps": len(results), "checkpoint": str(args.checkpoint), "history": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
