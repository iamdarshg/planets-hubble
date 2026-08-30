"""Bounded synthetic-pretrain and real-parent fine-tuning orchestration.

This module intentionally keeps the phases explicit.  Synthetic samples are
generated first; real Hubble parent exposures are accepted only through the
parented stream, and held-out evaluation is reported separately.  The runner
does not claim scientific convergence from a loss threshold: it records the
actual step history and stops at a configured budget or a repeated target-loss
criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch import Tensor, nn
import gc

from model import AstroMambaHConfig
from synthetic import RealObservationParent, SyntheticConfig

from .adapters import AstroMambaHTrainingAdapter, AstroMambaHTrainingBatch
from .harness import (
    DEFAULT_RSS_CAP_BYTES,
    DEFAULT_STORAGE_CAP_BYTES,
    BoundedTrainer,
    EpochReport,
    TrainingConfig,
    default_loss_fn,
    resolve_device,
)
from .synthetic import (
    iter_paired_synthetic_training_batches,
    iter_parented_synthetic_training_batches,
)


@dataclass(frozen=True)
class PhaseReport:
    phase: str
    reports: tuple[EpochReport, ...]
    stopped_reason: str
    checkpoint: str | None = None

    @property
    def losses(self) -> tuple[float, ...]:
        return tuple(report.last_loss for report in self.reports if report.last_loss is not None)


@dataclass(frozen=True)
class EvaluationReport:
    samples: int
    correct: int | None
    accuracy: float | None
    mean_event_probability: float
    predictions: tuple[float, ...] = field(default_factory=tuple)


def train_synthetic_then_real(
    *,
    model: nn.Module,
    synthetic_config: SyntheticConfig,
    real_parents: Iterable[RealObservationParent] = (),
    held_out_parents: Iterable[RealObservationParent] = (),
    device: torch.device | str = "auto",
    synthetic_max_steps: int = 32,
    real_max_steps: int = 16,
    target_loss: float = 0.05,
    target_patience: int = 3,
    output_dir: str | Path | None = None,
    rss_cap_bytes: int = DEFAULT_RSS_CAP_BYTES,
    storage_cap_bytes: int = DEFAULT_STORAGE_CAP_BYTES,
) -> dict[str, Any]:
    """Run the requested phases and return auditable reports.

    ``real_parents`` means real-parent injection/fine-tuning: the Hubble image,
    detector flags, cadence, and provenance come from the parent, while the
    event label is supplied by a controlled astrophysical injection.  A pure
    unlabeled real parent can still be scored with :func:`evaluate_unlabeled`.
    """
    if synthetic_max_steps < 1 or real_max_steps < 0:
        raise ValueError("training step budgets must be positive (real may be zero)")
    if target_loss <= 0.0 or target_patience < 1:
        raise ValueError("target_loss must be positive and target_patience must be >= 1")
    target_device = resolve_device(device)
    # AMP should reduce activation cost, not replace trainable master weights
    # with BF16. Direct BF16 SGD updates can quantize away small steps.
    model = model.to(target_device)
    trainer = BoundedTrainer(
        model,
        config=TrainingConfig(
            device=target_device,
            max_batches_per_epoch=1,
            amp="auto",
            grad_clip_norm=1.0,
            # Adam's two moment buffers are unnecessary for the bounded
            # lifecycle runner and can push host RSS over the local cap after
            # the first full-resolution step. SGD keeps the experiment's
            # optimizer state small enough to measure honestly.
            learning_rate=1e-4,
            rss_cap_bytes=rss_cap_bytes,
            storage_cap_bytes=storage_cap_bytes,
        ),
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-4),
    )

    synthetic_reports = _train_phase(
        trainer,
        "synthetic_pretraining",
        iter_paired_synthetic_training_batches,
        synthetic_config,
        synthetic_max_steps,
        target_loss,
        target_patience,
        real_parents=None,
    )
    synthetic_checkpoint = _save_checkpoint(trainer, output_dir, "synthetic_pretrained.pt", storage_cap_bytes)

    real_parent_tuple = tuple(real_parents)
    if real_parent_tuple and real_max_steps:
        real_reports = _train_phase(
            trainer,
            "real_parent_finetuning",
            iter_parented_synthetic_training_batches,
            None,
            real_max_steps,
            target_loss,
            target_patience,
            real_parents=real_parent_tuple,
        )
    else:
        real_reports = PhaseReport(
            phase="real_parent_finetuning",
            reports=(),
            stopped_reason="no real parent exposures supplied" if not real_parent_tuple else "real_max_steps=0",
        )
    real_checkpoint = _save_checkpoint(trainer, output_dir, "real_finetuned.pt", storage_cap_bytes)

    held_out_tuple = tuple(held_out_parents)
    held_out_eval = evaluate_parent_injections(
        trainer.model,
        held_out_tuple,
        device=target_device,
        sample_count=min(2, len(held_out_tuple)) if held_out_tuple else 0,
    )
    return {
        "device": str(target_device),
        "synthetic": synthetic_reports,
        "real": real_reports,
        "held_out_real_parent_evaluation": held_out_eval,
        "synthetic_checkpoint": synthetic_checkpoint,
        "real_checkpoint": real_checkpoint,
        "rss_cap_bytes": rss_cap_bytes,
        "storage_cap_bytes": storage_cap_bytes,
    }


def _train_phase(
    trainer: BoundedTrainer,
    phase: str,
    stream_factory: Any,
    config: SyntheticConfig | None,
    max_steps: int,
    target_loss: float,
    target_patience: int,
    *,
    real_parents: tuple[RealObservationParent, ...] | None,
) -> PhaseReport:
    reports: list[EpochReport] = []
    consecutive = 0
    for step in range(max_steps):
        if real_parents is None:
            stream: Iterator[AstroMambaHTrainingBatch] = stream_factory(
                config,
                sample_count=1,
                # Keep the counterfactual pair on CPU.  The runner splits it
                # before BoundedTrainer moves a single view to CUDA; retaining
                # a full paired GPU batch would defeat the memory bound.
                device="cpu",
            )
            pair_batch = next(stream)
            step_batches = _split_batch(pair_batch)
        else:
            step_batches = [next(stream_factory(
                real_parents,
                sample_count=1,
                device=trainer.device,
            ))]
        for batch in step_batches:
            report = trainer.train_epoch([batch], loss_fn=default_loss_fn)
            reports.append(report)
            if report.last_loss is not None and report.last_loss <= target_loss:
                consecutive += 1
                if consecutive >= target_patience:
                    return PhaseReport(phase, tuple(reports), "target_loss_reached")
            else:
                consecutive = 0
            if trainer.device.type == "cuda":
                del batch
                gc.collect()
                torch.cuda.synchronize(trainer.device)
                torch.cuda.empty_cache()
        if real_parents is None:
            del pair_batch, step_batches, stream
    return PhaseReport(phase, tuple(reports), "step_budget_exhausted")


def _split_batch(batch: AstroMambaHTrainingBatch) -> list[AstroMambaHTrainingBatch]:
    """Split a paired batch before the full-resolution model sees it.

    The paired generator still constructs null/injected views from one shared
    realization, but the 720x1280 activations are never duplicated on the GPU.
    """
    count = batch.batch_size
    if count < 2:
        return [batch]
    values: dict[str, Any] = {}
    for name in (
        "raster", "wavelength_tokens", "wavelength_mask", "object_tokens",
        "object_mask", "geometry", "exposure_duration", "coverage_vector",
        "local_time", "long_time", "coverage_map", "visit_mask", "step_mask",
        "source_xy",
    ):
        value = getattr(batch.inputs, name)
        values[name] = value
    result: list[AstroMambaHTrainingBatch] = []
    for index in range(count):
        input_values = {
            name: None if value is None else value[index:index + 1]
            for name, value in values.items()
        }
        result.append(
            AstroMambaHTrainingBatch(
                inputs=type(batch.inputs)(**input_values),
                target=batch.target[index:index + 1],
                auxiliary_targets={
                    name: value[index:index + 1]
                    for name, value in batch.auxiliary_targets.items()
                },
            )
        )
    return result


def evaluate_parent_injections(
    model: nn.Module,
    parents: Iterable[RealObservationParent],
    *,
    device: torch.device | str = "auto",
    sample_count: int,
    sequence_summary: bool = False,
) -> EvaluationReport:
    """Evaluate labeled injected/null counterfactuals from held-out parents."""
    parent_tuple = tuple(parents)
    if sample_count <= 0 or not parent_tuple:
        return EvaluationReport(0, None, None, 0.0)
    target_device = resolve_device(device)
    model = model.to(target_device)
    model.eval()
    predictions: list[float] = []
    labels: list[int] = []
    with torch.inference_mode():
        for batch in iter_parented_synthetic_training_batches(
            parent_tuple,
            sample_count=sample_count,
            device=target_device,
            sequence_summary=sequence_summary,
        ):
            with torch.autocast(
                device_type=target_device.type,
                dtype=torch.bfloat16 if target_device.type == "cuda" else torch.float32,
                enabled=target_device.type == "cuda",
            ):
                output = model(batch)
            probabilities = output["global_event_logits"].float().sigmoid().reshape(-1)
            predictions.extend(float(value) for value in probabilities.cpu())
            labels.extend(int(value) for value in batch.target.reshape(-1).cpu())
    predicted_labels = [int(value >= 0.5) for value in predictions]
    correct = sum(prediction == label for prediction, label in zip(predicted_labels, labels))
    return EvaluationReport(
        samples=len(labels),
        correct=correct,
        accuracy=correct / len(labels) if labels else None,
        mean_event_probability=sum(predictions) / len(predictions) if predictions else 0.0,
        predictions=tuple(predictions),
    )


def evaluate_unlabeled(
    model: nn.Module,
    batches: Iterable[AstroMambaHTrainingBatch],
    *,
    device: torch.device | str = "auto",
) -> EvaluationReport:
    """Score real images without inventing a ground-truth detection label."""
    target_device = resolve_device(device)
    model = model.to(target_device)
    model.eval()
    predictions: list[float] = []
    with torch.inference_mode():
        for batch in batches:
            with torch.autocast(
                device_type=target_device.type,
                dtype=torch.bfloat16 if target_device.type == "cuda" else torch.float32,
                enabled=target_device.type == "cuda",
            ):
                output = model(batch.to(target_device))
            predictions.extend(float(value) for value in output["global_event_logits"].sigmoid().reshape(-1).cpu())
    return EvaluationReport(
        samples=len(predictions),
        correct=None,
        accuracy=None,
        mean_event_probability=sum(predictions) / len(predictions) if predictions else 0.0,
        predictions=tuple(predictions),
    )


def _save_checkpoint(
    trainer: BoundedTrainer,
    output_dir: str | Path | None,
    filename: str,
    storage_cap_bytes: int,
) -> str | None:
    if output_dir is None:
        return None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    torch.save(
        {
            "model": trainer.model.state_dict(),
            "training_state": trainer.state.__dict__,
            "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters()),
        },
        path,
    )
    total = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
    if total > storage_cap_bytes:
        raise RuntimeError(f"checkpoint storage cap exceeded: {total} > {storage_cap_bytes}")
    return str(path)
