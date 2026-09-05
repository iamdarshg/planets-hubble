"""Automate bounded synthetic curriculum probes before fixed real holdout audit.

Each synthetic stage runs in a separate child process.  That keeps RSS bounded
per stage, preserves each checkpoint/report, and prevents a bad stage from
silently replacing the best candidate.  The fixed real holdout is evaluation
only: this runner audits it only after a synthetic stage reaches the requested
synthetic error gate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    field_star_count: int
    field_planet_probability: float
    stellar_brightness_noise_sigma: float
    transit_radius_ratio_min: float
    transit_radius_ratio_max: float
    visits: int
    local_steps: int
    visit_spacing_days: float
    transit_period_days: float
    learning_rate: float
    negative_loss_weight: float
    paired_ranking_weight: float
    auxiliary_paired_ranking_weight: float
    max_epochs: int


def _default_stages() -> list[Stage]:
    return [
        Stage(
            name="target-signal",
            field_star_count=4,
            field_planet_probability=0.0,
            stellar_brightness_noise_sigma=0.0003,
            transit_radius_ratio_min=0.045,
            transit_radius_ratio_max=0.09,
            visits=3,
            local_steps=16,
            visit_spacing_days=3.0,
            transit_period_days=3.0,
            learning_rate=3.0e-4,
            negative_loss_weight=1.0,
            paired_ranking_weight=0.25,
            auxiliary_paired_ranking_weight=0.02,
            max_epochs=4,
        ),
        Stage(
            name="mild-crowding",
            field_star_count=6,
            field_planet_probability=0.02,
            stellar_brightness_noise_sigma=0.0005,
            transit_radius_ratio_min=0.03,
            transit_radius_ratio_max=0.08,
            visits=3,
            local_steps=16,
            visit_spacing_days=3.0,
            transit_period_days=3.0,
            learning_rate=1.0e-4,
            negative_loss_weight=1.0,
            paired_ranking_weight=0.5,
            auxiliary_paired_ranking_weight=0.02,
            max_epochs=4,
        ),
        Stage(
            name="realistic-crowding",
            field_star_count=8,
            field_planet_probability=0.10,
            stellar_brightness_noise_sigma=0.0008,
            transit_radius_ratio_min=0.012,
            transit_radius_ratio_max=0.05,
            visits=4,
            local_steps=16,
            visit_spacing_days=3.0,
            transit_period_days=3.0,
            learning_rate=5.0e-5,
            negative_loss_weight=1.1,
            paired_ranking_weight=0.5,
            auxiliary_paired_ranking_weight=0.02,
            max_epochs=4,
        ),
        Stage(
            name="hard-crowding",
            field_star_count=8,
            field_planet_probability=0.45,
            stellar_brightness_noise_sigma=0.0015,
            transit_radius_ratio_min=0.006,
            transit_radius_ratio_max=0.04,
            visits=4,
            local_steps=16,
            visit_spacing_days=2.7,
            transit_period_days=2.7,
            learning_rate=2.0e-5,
            negative_loss_weight=1.2,
            paired_ranking_weight=0.5,
            auxiliary_paired_ranking_weight=0.02,
            max_epochs=4,
        ),
    ]


def _run_child(command: list[str], *, cwd: Path) -> int:
    print(json.dumps({"command": command}, sort_keys=True), flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    return int(completed.returncode)


def _load_report(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"expected report was not written: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _holdout_errors(report: dict[str, object]) -> int:
    holdout = report.get("holdout")
    if not isinstance(holdout, dict) or holdout.get("errors") is None:
        raise RuntimeError("synthetic report is missing holdout errors")
    return int(holdout["errors"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-checkpoint", type=Path)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=128)
    parser.add_argument("--holdout-pairs", type=int, default=128)
    parser.add_argument("--batch-pairs", type=int, default=32)
    parser.add_argument("--stage-count", type=int, default=4)
    parser.add_argument(
        "--min-stage-epochs",
        type=int,
        default=None,
        help="Raise every curriculum stage to at least this many epochs.",
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rss-cap-bytes", type=int, default=1_200_000_000)
    parser.add_argument("--synthetic-error-gate", type=int, default=3)
    parser.add_argument("--real-loss-gate", type=float, default=0.20)
    parser.add_argument(
        "--loss-mode",
        choices=("event", "source"),
        default="source",
        help="Synthetic trainer objective; source mode adds direct source-event supervision.",
    )
    parser.add_argument(
        "--auxiliary-event-head-weight",
        type=float,
        default=0.25,
        help="BCE weight for directly supervising compact event evidence heads.",
    )
    parser.add_argument("--real-batch-size", type=int, default=8)
    parser.add_argument("--aperture-fraction", type=float, default=0.125)
    parser.add_argument("--start-index", type=int, default=5_000_000)
    parser.add_argument("--holdout-start-index", type=int, default=6_000_000)
    parser.add_argument("--reset-source-photometry-branch", action="store_true")
    parser.add_argument("--reset-source-tokenizer", action="store_true")
    parser.add_argument("--reset-temporal-event-heads", action="store_true")
    parser.add_argument("--zero-event-photometry-weight", action="store_true")
    parser.add_argument("--initialize-source-dip-event-prior", action="store_true")
    parser.add_argument("--train-only-event-heads", action="store_true")
    parser.add_argument("--train-source-tokenizer-with-event-heads", action="store_true")
    parser.add_argument("--train-only-event-calibration", action="store_true")
    parser.add_argument("--defer-training-eval-until-final", action="store_true")
    parser.add_argument("--training-eval-frequency", type=int, default=1)
    parser.add_argument("--progress-log-frequency", type=int, default=0)
    args = parser.parse_args()

    if args.cycles < 1 or args.stage_count < 1:
        raise ValueError("cycles and stage-count must be positive")
    if args.min_stage_epochs is not None and args.min_stage_epochs < 1:
        raise ValueError("min-stage-epochs must be positive")
    if min(args.pair_count, args.holdout_pairs, args.batch_pairs, args.real_batch_size) < 1:
        raise ValueError("batch and pair counts must be positive")
    if args.rss_cap_bytes < 1:
        raise ValueError("rss-cap-bytes must be positive")
    if args.synthetic_error_gate < 0:
        raise ValueError("synthetic-error-gate must be non-negative")
    if args.real_loss_gate <= 0.0:
        raise ValueError("real-loss-gate must be positive")
    if args.auxiliary_event_head_weight < 0.0:
        raise ValueError("auxiliary-event-head-weight must be non-negative")
    if args.training_eval_frequency < 1:
        raise ValueError("training-eval-frequency must be positive")
    if args.progress_log_frequency < 0:
        raise ValueError("progress-log-frequency must be non-negative")
    if not args.from_scratch and (args.input_checkpoint is None or not args.input_checkpoint.is_file()):
        raise FileNotFoundError(args.input_checkpoint)
    if args.from_scratch and args.input_checkpoint is not None:
        raise ValueError("--from-scratch and --input-checkpoint are mutually exclusive")

    repo = Path(__file__).resolve().parents[1]
    trainer = Path(__file__).with_name("train_synthetic_until_perfect.py")
    fixed_evaluator = Path(__file__).with_name("evaluate_fixed_real_holdout.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stages = _default_stages()[: args.stage_count]
    current_checkpoint = args.input_checkpoint.resolve() if args.input_checkpoint else None
    best_synthetic_checkpoint: Path | None = current_checkpoint
    best_synthetic_errors: int | None = None
    best_real_checkpoint: Path | None = None
    best_real_loss: float | None = None
    history: list[dict[str, object]] = []
    started = time.time()

    for cycle in range(args.cycles):
        for stage_index, stage in enumerate(stages):
            ordinal = cycle * len(stages) + stage_index
            stage_slug = f"cycle{cycle + 1:02d}-stage{stage_index + 1:02d}-{stage.name}"
            output_checkpoint = args.output_dir / f"{stage_slug}.pt"
            cache_dir = args.output_dir / f"cache-{stage_slug}"
            start_index = args.start_index + ordinal * max(args.pair_count, 1)
            holdout_start_index = args.holdout_start_index + ordinal * max(args.holdout_pairs, 1)
            command = [
                sys.executable,
                str(trainer),
                "--output-checkpoint",
                str(output_checkpoint),
                "--pair-count",
                str(args.pair_count),
                "--start-index",
                str(start_index),
                "--holdout-pairs",
                str(args.holdout_pairs),
                "--holdout-start-index",
                str(holdout_start_index),
                "--batch-pairs",
                str(args.batch_pairs),
                "--max-epochs",
                str(max(stage.max_epochs, args.min_stage_epochs or stage.max_epochs)),
                "--learning-rate",
                str(stage.learning_rate),
                "--loss-mode",
                args.loss_mode,
                "--positive-loss-weight",
                "1.0",
                "--negative-loss-weight",
                str(stage.negative_loss_weight),
                "--paired-ranking-weight",
                str(stage.paired_ranking_weight),
                "--paired-ranking-margin",
                "0.5",
                "--auxiliary-event-head-weight",
                str(args.auxiliary_event_head_weight),
                "--auxiliary-paired-ranking-weight",
                str(stage.auxiliary_paired_ranking_weight),
                "--training-eval-frequency",
                str(args.training_eval_frequency),
                "--progress-log-frequency",
                str(args.progress_log_frequency),
                "--field-star-count",
                str(stage.field_star_count),
                "--field-planet-probability",
                str(stage.field_planet_probability),
                "--stellar-brightness-noise-sigma",
                str(stage.stellar_brightness_noise_sigma),
                "--transit-radius-ratio-min",
                str(stage.transit_radius_ratio_min),
                "--transit-radius-ratio-max",
                str(stage.transit_radius_ratio_max),
                "--visits",
                str(stage.visits),
                "--local-steps",
                str(stage.local_steps),
                "--visit-spacing-days",
                str(stage.visit_spacing_days),
                "--transit-period-days",
                str(stage.transit_period_days),
                "--device",
                args.device,
                "--seed",
                str(args.seed + ordinal),
                "--cache-dir",
                str(cache_dir),
                "--rss-cap-bytes",
                str(args.rss_cap_bytes),
                "--stop-holdout-errors",
                str(args.synthetic_error_gate),
            ]
            if current_checkpoint is None:
                command.append("--from-scratch")
            else:
                command.extend(["--input-checkpoint", str(current_checkpoint)])
            if args.reset_source_photometry_branch and ordinal == 0:
                command.append("--reset-source-photometry-branch")
            if args.reset_source_tokenizer and ordinal == 0:
                command.append("--reset-source-tokenizer")
            if args.reset_temporal_event_heads and ordinal == 0:
                command.append("--reset-temporal-event-heads")
            if args.zero_event_photometry_weight and ordinal == 0:
                command.append("--zero-event-photometry-weight")
            if args.initialize_source_dip_event_prior and ordinal == 0:
                command.append("--initialize-source-dip-event-prior")
            if args.train_only_event_heads:
                command.append("--train-only-event-heads")
            if args.train_source_tokenizer_with_event_heads:
                command.append("--train-source-tokenizer-with-event-heads")
            if args.train_only_event_calibration:
                command.append("--train-only-event-calibration")
            if args.defer_training_eval_until_final:
                command.append("--defer-training-eval-until-final")

            exit_code = _run_child(command, cwd=repo)
            report_path = output_checkpoint.with_suffix(".report.json")
            report = _load_report(report_path)
            errors = _holdout_errors(report)
            synthetic_improved = best_synthetic_errors is None or errors < best_synthetic_errors
            if synthetic_improved:
                best_synthetic_errors = errors
                best_synthetic_checkpoint = output_checkpoint.resolve()
            if synthetic_improved or errors <= args.synthetic_error_gate:
                current_checkpoint = output_checkpoint.resolve()
                advanced_checkpoint = output_checkpoint.resolve()
            else:
                current_checkpoint = best_synthetic_checkpoint
                advanced_checkpoint = best_synthetic_checkpoint
            entry: dict[str, object] = {
                "cycle": cycle + 1,
                "stage": stage.name,
                "checkpoint": str(output_checkpoint.resolve()),
                "advanced_checkpoint": str(advanced_checkpoint) if advanced_checkpoint else None,
                "synthetic_exit_code": exit_code,
                "synthetic_status": report.get("status"),
                "synthetic_holdout": report.get("holdout"),
                "synthetic_training": report.get("training"),
                "synthetic_improved": synthetic_improved,
                "loss_mode": args.loss_mode,
                "auxiliary_event_head_weight": args.auxiliary_event_head_weight,
                "train_only_event_calibration": args.train_only_event_calibration,
                "train_source_tokenizer_with_event_heads": args.train_source_tokenizer_with_event_heads,
                "defer_training_eval_until_final": args.defer_training_eval_until_final,
                "training_eval_frequency": args.training_eval_frequency,
                "progress_log_frequency": args.progress_log_frequency,
                "visits": stage.visits,
                "local_steps": stage.local_steps,
                "visit_spacing_days": stage.visit_spacing_days,
                "transit_period_days": stage.transit_period_days,
                "best_synthetic_checkpoint": str(best_synthetic_checkpoint) if best_synthetic_checkpoint else None,
                "best_synthetic_errors": best_synthetic_errors,
                "rss_cap_bytes": args.rss_cap_bytes,
                "peak_process_rss_bytes": report.get("peak_process_rss_bytes"),
            }

            if errors <= args.synthetic_error_gate:
                fixed_output = args.output_dir / f"{stage_slug}-fixed-real-holdout.json"
                fixed_command = [
                    sys.executable,
                    str(fixed_evaluator),
                    "--manifest",
                    str(args.manifest),
                    "--checkpoint",
                    str(output_checkpoint),
                    "--output",
                    str(fixed_output),
                    "--batch-size",
                    str(args.real_batch_size),
                    "--device",
                    args.device,
                    "--aperture-fraction",
                    str(args.aperture_fraction),
                ]
                fixed_exit_code = _run_child(fixed_command, cwd=repo)
                fixed_report = _load_report(fixed_output)
                metrics = fixed_report.get("metrics")
                real_loss = (
                    float(metrics["mean_bce_loss"])
                    if isinstance(metrics, dict) and metrics.get("mean_bce_loss") is not None
                    else None
                )
                if real_loss is not None and (
                    best_real_loss is None or real_loss < best_real_loss
                ):
                    best_real_loss = real_loss
                    best_real_checkpoint = output_checkpoint.resolve()
                entry.update(
                    {
                        "fixed_real_exit_code": fixed_exit_code,
                        "fixed_real_report": str(fixed_output),
                        "fixed_real_status": fixed_report.get("status"),
                        "fixed_real_metrics": metrics,
                        "best_real_checkpoint": str(best_real_checkpoint) if best_real_checkpoint else None,
                        "best_real_loss": best_real_loss,
                    }
                )
                history.append(entry)
                print(json.dumps(entry, sort_keys=True), flush=True)
                if real_loss is not None and real_loss < args.real_loss_gate:
                    summary = {
                        "status": "pass",
                        "best_checkpoint": str(output_checkpoint.resolve()),
                        "best_real_loss": real_loss,
                        "best_synthetic_errors": best_synthetic_errors,
                        "history": history,
                        "elapsed_seconds": time.time() - started,
                    }
                    summary_path = args.output_dir / "synthetic-curriculum-summary.json"
                    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    print(json.dumps(summary, sort_keys=True), flush=True)
                    return 0
            else:
                entry["fixed_real_evaluation"] = "skipped_until_synthetic_gate_passes"
                history.append(entry)
                print(json.dumps(entry, sort_keys=True), flush=True)

    summary = {
        "status": "synthetic_gate_not_met",
        "best_synthetic_checkpoint": str(best_synthetic_checkpoint) if best_synthetic_checkpoint else None,
        "best_synthetic_errors": best_synthetic_errors,
        "best_real_checkpoint": str(best_real_checkpoint) if best_real_checkpoint else None,
        "best_real_loss": best_real_loss,
        "synthetic_error_gate": args.synthetic_error_gate,
        "real_loss_gate": args.real_loss_gate,
        "fixed_real_evaluation": "only run for stages passing the synthetic gate",
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    summary_path = args.output_dir / "synthetic-curriculum-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if best_real_loss is not None and best_real_loss < args.real_loss_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
