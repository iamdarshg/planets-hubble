"""Automatically continue bounded real-data fine-tuning by validation BCE.

Each iteration is a separate one-epoch process. That keeps a failed epoch from
destroying the best checkpoint and bounds memory independently for every
iteration. The fixed 100-example test holdout is deliberately not consulted
by this loop; it remains an evaluation-only gate after the loop finishes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=5.0e-4)
    parser.add_argument("--target-validation-bce", type=float, default=0.20)
    parser.add_argument("--launch-retries", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument(
        "--loss-mode",
        choices=("event", "mil"),
        default="event",
        help="training objective passed to each one-epoch child",
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rss-cap-bytes", type=int, default=1_200_000_000)
    parser.add_argument(
        "--reset-source-photometry-branch",
        action="store_true",
        help="Reinitialize the source-photometry event branch before each child epoch.",
    )
    parser.add_argument(
        "--train-only-temporal-summary",
        action="store_true",
        help="Freeze the representation and train only the temporal summary calibration head.",
    )
    args = parser.parse_args()
    if args.max_epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.launch_retries < 0:
        raise ValueError("max-epochs, patience, and batch-size must be positive; launch-retries cannot be negative")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning-rate must be positive and weight-decay cannot be negative")
    if args.min_delta < 0.0 or args.target_validation_bce <= 0.0 or args.rss_cap_bytes < 1:
        raise ValueError("min-delta cannot be negative; target and RSS cap must be positive")

    repo = Path(__file__).resolve().parents[1]
    trainer = Path(__file__).with_name("train_kepler_real.py")
    input_checkpoint = args.input_checkpoint.resolve()
    if not input_checkpoint.is_file():
        raise FileNotFoundError(input_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint = input_checkpoint
    best_bce = float("inf")
    stale_epochs = 0
    history: list[dict[str, object]] = []
    start_time = time.time()
    for iteration in range(args.max_epochs):
        output_checkpoint = args.output_dir / f"real-continuation-epoch-{iteration + 1:03d}.pt"
        command = [
            sys.executable,
            str(trainer),
            "--manifest",
            str(args.manifest),
            "--input-checkpoint",
            str(best_checkpoint),
            "--output-checkpoint",
            str(output_checkpoint),
            "--batch-size",
            str(args.batch_size),
            "--epochs",
            "1",
            "--learning-rate",
            str(args.learning_rate),
            "--optimizer",
            "adamw",
            "--scheduler",
            "cosine",
            "--loss-mode",
            args.loss_mode,
            "--weight-decay",
            str(args.weight_decay),
            "--seed",
            str(args.seed + iteration),
            "--device",
            args.device,
            "--rss-cap-bytes",
            str(args.rss_cap_bytes),
        ]
        if args.reset_source_photometry_branch and iteration == 0:
            command.append("--reset-source-photometry-branch")
        if args.train_only_temporal_summary:
            command.append("--train-only-temporal-summary")
        print(json.dumps({"iteration": iteration + 1, "input_checkpoint": str(best_checkpoint), "command": command}), flush=True)
        completed = None
        for attempt in range(args.launch_retries + 1):
            completed = subprocess.run(command, cwd=repo, check=False)
            if completed.returncode == 0:
                break
            if attempt >= args.launch_retries:
                raise RuntimeError(
                    f"real training iteration {iteration + 1} failed with exit code {completed.returncode}"
                )
            delay = min(300, 30 * (2**attempt))
            print(
                json.dumps(
                    {
                        "iteration": iteration + 1,
                        "attempt": attempt + 1,
                        "status": "retrying_child_start",
                        "exit_code": completed.returncode,
                        "delay_seconds": delay,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(delay)
        if completed is None or completed.returncode != 0:
            raise RuntimeError(f"real training iteration {iteration + 1} did not complete")
        report_path = output_checkpoint.with_suffix(".report.json")
        if not report_path.is_file():
            raise RuntimeError(f"training completed without report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # train_kepler_real records per-epoch metrics under ``history``;
        # retain the runner's validation-BCE selection rule without relying
        # on a non-existent top-level ``validation`` field.
        history_report = report.get("history")
        validation = history_report[-1].get("validation") if isinstance(history_report, list) and history_report else None
        if not isinstance(validation, dict) or validation.get("mean_bce_loss") is None:
            raise RuntimeError(f"report has no validation BCE: {report_path}")
        validation_bce = float(validation["mean_bce_loss"])
        improved = validation_bce < best_bce - args.min_delta
        if improved:
            best_bce = validation_bce
            best_checkpoint = output_checkpoint.resolve()
            stale_epochs = 0
        else:
            stale_epochs += 1
        entry = {
            "iteration": iteration + 1,
            "checkpoint": str(output_checkpoint.resolve()),
            "validation_bce": validation_bce,
            "improved": improved,
            "best_checkpoint": str(best_checkpoint),
            "best_validation_bce": best_bce,
            "stale_epochs": stale_epochs,
            "peak_process_rss_bytes": report.get("peak_process_rss_bytes"),
            "rss_cap_bytes": args.rss_cap_bytes,
        }
        history.append(entry)
        print(json.dumps(entry, sort_keys=True), flush=True)
        if best_bce <= args.target_validation_bce or stale_epochs >= args.patience:
            break

    summary = {
        "status": "complete",
        "best_checkpoint": str(best_checkpoint),
        "best_validation_bce": best_bce,
        "iterations": len(history),
        "history": history,
        "target_validation_bce": args.target_validation_bce,
        "fixed_holdout_evaluation": "not run by this loop",
        "elapsed_seconds": time.time() - start_time,
    }
    summary_path = args.output_dir / "real-continuation-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "best_checkpoint": summary["best_checkpoint"], "best_validation_bce": best_bce, "iterations": len(history)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
