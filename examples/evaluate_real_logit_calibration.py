"""Fit validation-only logit calibration and audit the fixed real holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

EXAMPLES = Path(__file__).resolve().parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from train_kepler_real import _batch_records, _load_model, _make_batch, _records  # noqa: E402


def _collect_logits(
    model,
    root: Path,
    records: list[dict[str, object]],
    device: torch.device,
    batch_size: int,
    *,
    aperture_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch_records in _batch_records(records, batch_size, seed=0, shuffle=False):
            batch = _make_batch(
                root,
                batch_records,
                device,
                aperture_fraction=aperture_fraction,
            )
            output = model(batch)
            logits.append(output["global_event_logits"].float().reshape(-1).cpu())
            labels.append(batch.target.float().reshape(-1).cpu())
            del batch, output
    return torch.cat(logits), torch.cat(labels)


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int]:
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probabilities = logits.sigmoid()
    predictions = probabilities >= 0.5
    truth = labels >= 0.5
    correct = int((predictions == truth).sum().item())
    return {
        "samples": int(labels.numel()),
        "correct": correct,
        "accuracy": correct / max(int(labels.numel()), 1),
        "mean_bce_loss": float(losses.mean().item()),
        "mean_probability": float(probabilities.mean().item()),
        "true_positive": int((predictions & truth).sum().item()),
        "true_negative": int((~predictions & ~truth).sum().item()),
        "false_positive": int((predictions & ~truth).sum().item()),
        "false_negative": int((~predictions & truth).sum().item()),
    }


def _rank_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | int | None]:
    positive = logits[labels >= 0.5]
    negative = logits[labels < 0.5]
    if positive.numel() == 0 or negative.numel() == 0:
        return {"auc": None, "positive_count": int(positive.numel()), "negative_count": int(negative.numel())}
    comparisons = positive[:, None] - negative[None, :]
    wins = (comparisons > 0).float().sum()
    ties = (comparisons == 0).float().sum()
    auc = (wins + 0.5 * ties) / comparisons.numel()
    return {
        "auc": float(auc.item()),
        "positive_count": int(positive.numel()),
        "negative_count": int(negative.numel()),
        "positive_mean_logit": float(positive.mean().item()),
        "negative_mean_logit": float(negative.mean().item()),
    }


def _fit_affine(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    raw_scale = torch.zeros((), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([raw_scale, bias], lr=0.25, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        scale = F.softplus(raw_scale) + 1.0e-4
        loss = F.binary_cross_entropy_with_logits(scale * logits + bias, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return float((F.softplus(raw_scale) + 1.0e-4).item()), float(bias.item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--aperture-fraction", type=float, default=0.125)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not 0.0 < args.aperture_fraction <= 1.0:
        raise ValueError("aperture-fraction must be in (0, 1]")

    device = torch.device(args.device)
    records = _records(args.manifest)
    validation = [record for record in records if record.get("split") == "validation"]
    test = [record for record in records if record.get("split") == "test"][:100]
    if not validation or len(test) != 100:
        raise ValueError("manifest must contain validation records and at least 100 test records")
    model = _load_model(args.checkpoint, device)
    validation_logits, validation_labels = _collect_logits(
        model,
        args.manifest.parent,
        validation,
        device,
        args.batch_size,
        aperture_fraction=args.aperture_fraction,
    )
    fixed_logits, fixed_labels = _collect_logits(
        model,
        args.manifest.parent,
        test,
        device,
        args.batch_size,
        aperture_fraction=args.aperture_fraction,
    )
    scale, bias = _fit_affine(validation_logits, validation_labels)
    report = {
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "calibration_fit": "affine sigmoid calibration fitted on validation split only",
        "scale": scale,
        "bias": bias,
        "validation_raw": _metrics(validation_logits, validation_labels),
        "validation_calibrated": _metrics(validation_logits * scale + bias, validation_labels),
        "validation_rank": _rank_metrics(validation_logits, validation_labels),
        "fixed_holdout_raw": _metrics(fixed_logits, fixed_labels),
        "fixed_holdout_calibrated": _metrics(fixed_logits * scale + bias, fixed_labels),
        "fixed_holdout_rank": _rank_metrics(fixed_logits, fixed_labels),
        "fixed_holdout_gate": (
            "pass"
            if _metrics(fixed_logits * scale + bias, fixed_labels)["mean_bce_loss"] < 0.20
            else "fail"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
