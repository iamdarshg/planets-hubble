"""Probe synthetic event-head separation for a saved compact checkpoint.

This diagnostic intentionally reuses the cap-safe synthetic training adapter and
batch construction path from ``train_synthetic_until_perfect.py``.  It reports
which event evidence heads separate injected target transits from their matched
null views before we spend more CPU on a full curriculum run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_synthetic_until_perfect import (  # noqa: E402
    _apply_curriculum_overrides,
    _batch_cache_path,
    _load_model,
    _make_batch,
    _rss,
    _synthetic_config,
)


HEADS = (
    "global_event_logits",
    "source_photometry_event_logits",
    "source_event_logits",
    "pooled_backbone_event_logits",
    "temporal_summary_event_logits",
    "temporal_shape_event_logits",
    "temporal_robust_event_logits",
    "temporal_matched_event_logits",
    "temporal_sequence_event_logits",
    "temporal_multiscale_event_logits",
    "temporal_feature_fusion_event_logits",
    "source_dip_event_logits",
)


def _flatten_head(output: dict[str, torch.Tensor], name: str) -> torch.Tensor | None:
    value = output.get(name)
    if not isinstance(value, torch.Tensor):
        return None
    if value.ndim == 2 and value.shape[1] >= 1:
        value = value[:, 0]
    return value.reshape(-1).detach().float().cpu()


def _auc(probabilities: list[float], labels: list[int]) -> float | None:
    positives = [score for score, label in zip(probabilities, labels) if label == 1]
    negatives = [score for score, label in zip(probabilities, labels) if label == 0]
    total = len(positives) * len(negatives)
    if total == 0:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total


def _summarize(probabilities: list[float], labels: list[int]) -> dict[str, float | int | None]:
    predicted = [1 if probability >= 0.5 else 0 for probability in probabilities]
    positives = [score for score, label in zip(probabilities, labels) if label == 1]
    negatives = [score for score, label in zip(probabilities, labels) if label == 0]
    correct = sum(int(prediction == label) for prediction, label in zip(predicted, labels))
    pair_wins = 0
    pair_total = len(labels) // 2
    for index in range(pair_total):
        null_probability = probabilities[index * 2]
        injected_probability = probabilities[index * 2 + 1]
        pair_wins += int(injected_probability > null_probability)
    return {
        "samples": len(labels),
        "correct_at_0_5": correct,
        "errors_at_0_5": len(labels) - correct,
        "accuracy_at_0_5": correct / len(labels) if labels else None,
        "pair_ranking_wins": pair_wins,
        "pair_ranking_errors": pair_total - pair_wins,
        "pair_ranking_accuracy": pair_wins / pair_total if pair_total else None,
        "auc": _auc(probabilities, labels),
        "mean_positive_probability": sum(positives) / len(positives) if positives else None,
        "mean_negative_probability": sum(negatives) / len(negatives) if negatives else None,
        "mean_probability": sum(probabilities) / len(probabilities) if probabilities else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=256)
    parser.add_argument("--batch-pairs", type=int, default=48)
    parser.add_argument("--start-index", type=int, default=14_000_000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--field-star-count", type=int, default=6)
    parser.add_argument("--field-planet-probability", type=float, default=0.02)
    parser.add_argument("--stellar-brightness-noise-sigma", type=float, default=0.0005)
    parser.add_argument("--transit-radius-ratio-min", type=float, default=0.03)
    parser.add_argument("--transit-radius-ratio-max", type=float, default=0.08)
    parser.add_argument("--rss-cap-bytes", type=int, default=1_200_000_000)
    args = parser.parse_args()

    if min(args.pair_count, args.batch_pairs) < 1:
        raise ValueError("pair-count and batch-pairs must be positive")
    if args.rss_cap_bytes < 1:
        raise ValueError("rss-cap-bytes must be positive")

    device = torch.device(args.device)
    model = _load_model(args.checkpoint, device)
    model.eval()
    generator_config = _apply_curriculum_overrides(
        _synthetic_config(seed=args.seed),
        field_star_count=args.field_star_count,
        field_planet_probability=args.field_planet_probability,
        stellar_brightness_noise_sigma=args.stellar_brightness_noise_sigma,
    )
    cache_dir = args.cache_dir or args.output.parent / "probe-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    labels: list[int] = []
    by_head: dict[str, list[float]] = {name: [] for name in HEADS}
    peak_rss = _rss()

    with torch.inference_mode():
        for offset in range(0, args.pair_count, args.batch_pairs):
            count = min(args.batch_pairs, args.pair_count - offset)
            start_index = args.start_index + offset
            batch = _make_batch(
                generator_config,
                start_index,
                count,
                device,
                _batch_cache_path(cache_dir, start_index, count),
                transit_radius_ratio_min=args.transit_radius_ratio_min,
                transit_radius_ratio_max=args.transit_radius_ratio_max,
            )
            output = model(batch)
            labels.extend(int(value) for value in batch.target.reshape(-1).detach().cpu().tolist())
            for name in HEADS:
                logits = _flatten_head(output, name)
                if logits is not None:
                    by_head[name].extend(float(value) for value in logits.sigmoid().tolist())
            peak_rss = max(peak_rss, _rss())
            if peak_rss > args.rss_cap_bytes:
                raise RuntimeError(f"RSS cap exceeded: {peak_rss} > {args.rss_cap_bytes}")
            del batch, output

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "pair_count": args.pair_count,
        "batch_pairs": args.batch_pairs,
        "start_index": args.start_index,
        "device": str(device),
        "rss_cap_bytes": args.rss_cap_bytes,
        "peak_process_rss_bytes": peak_rss,
        "generator_config": {
            key: value
            for key, value in asdict(generator_config).items()
            if key
            in {
                "field_star_count",
                "field_planet_probability",
                "stellar_brightness_noise_sigma",
                "stellar_brightness_ar1",
                "stellar_brightness_amplitude_scatter",
                "raster_height",
                "raster_width",
                "local_steps",
            }
        },
        "heads": {
            name: _summarize(probabilities, labels)
            for name, probabilities in by_head.items()
            if len(probabilities) == len(labels)
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
