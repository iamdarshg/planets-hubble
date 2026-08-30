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
import inspect
import json
import os
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

DEFAULT_SYNTHETIC_MIN_EXAMPLES = 4096
DEFAULT_SYNTHETIC_CACHE_DIR = Path(
    os.environ.get(
        "PLANETS_HUBBLE_SYNTHETIC_CACHE_DIR",
        "artifacts/synthetic-cache",
    )
)
DEFAULT_SYNTHETIC_CACHE_SIZE_MIB = 2048


@dataclass(frozen=True)
class PhaseReport:
    phase: str
    reports: tuple[EpochReport, ...]
    stopped_reason: str
    checkpoint: str | None = None

    @property
    def losses(self) -> tuple[float, ...]:
        return tuple(report.last_loss for report in self.reports if report.last_loss is not None)

    @property
    def examples_seen(self) -> int:
        """Return the number of examples actually consumed in this phase."""
        return sum(report.samples_seen for report in self.reports)


@dataclass(frozen=True)
class CurriculumGateReport:
    """Audit record for admission from synthetic to real-parent training."""

    synthetic_examples_seen: int
    minimum_synthetic_examples: int
    is_open: bool
    reason: str


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
    synthetic_max_steps: int = DEFAULT_SYNTHETIC_MIN_EXAMPLES // 2,
    synthetic_min_examples: int = DEFAULT_SYNTHETIC_MIN_EXAMPLES,
    bounded_smoke_test: bool = False,
    synthetic_cache_dir: str | Path | None = DEFAULT_SYNTHETIC_CACHE_DIR,
    synthetic_cache_size: int = DEFAULT_SYNTHETIC_CACHE_SIZE_MIB,
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
    The optional synthetic cache is disk-only: its default is an SSD-oriented
    D: path, and its integer size is a MiB budget rather than an in-RAM count.
    """
    if synthetic_max_steps < 1 or real_max_steps < 0:
        raise ValueError("training step budgets must be positive (real may be zero)")
    if synthetic_min_examples < 1:
        raise ValueError("synthetic_min_examples must be positive")
    if synthetic_min_examples < DEFAULT_SYNTHETIC_MIN_EXAMPLES and not bounded_smoke_test:
        raise ValueError(
            "synthetic_min_examples below the production minimum requires "
            "bounded_smoke_test=True"
        )
    if synthetic_cache_size < 1:
        raise ValueError("synthetic_cache_size must be positive")
    synthetic_cache_budget_bytes = synthetic_cache_size * 1024 * 1024
    if synthetic_cache_budget_bytes > storage_cap_bytes:
        raise ValueError("synthetic cache budget exceeds storage cap")
    synthetic_cache_path = _prepare_synthetic_cache_dir(synthetic_cache_dir)
    synthetic_cache_stream_supported = _stream_supports_synthetic_cache(
        iter_paired_synthetic_training_batches
    )
    cache_bytes_before_training = _directory_bytes(synthetic_cache_path)
    if cache_bytes_before_training > synthetic_cache_budget_bytes:
        raise RuntimeError(
            "synthetic cache exceeds configured budget: "
            f"{cache_bytes_before_training} > {synthetic_cache_budget_bytes}"
        )
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
        synthetic_cache_dir=synthetic_cache_path,
        synthetic_cache_size=synthetic_cache_size,
        minimum_examples=synthetic_min_examples,
    )
    synthetic_cache_bytes = _directory_bytes(synthetic_cache_path)
    if synthetic_cache_bytes > synthetic_cache_budget_bytes:
        raise RuntimeError(
            "synthetic cache exceeds configured budget: "
            f"{synthetic_cache_bytes} > {synthetic_cache_budget_bytes}"
        )
    synthetic_checkpoint = _save_checkpoint(
        trainer,
        output_dir,
        "synthetic_pretrained.pt",
        storage_cap_bytes,
        reserved_storage_bytes=synthetic_cache_bytes,
    )
    real_phase_gate = _synthetic_curriculum_gate(
        synthetic_reports.examples_seen,
        synthetic_min_examples,
    )

    real_parent_tuple = tuple(real_parents)
    if not real_phase_gate.is_open:
        real_reports = PhaseReport(
            phase="real_parent_finetuning",
            reports=(),
            stopped_reason=real_phase_gate.reason,
        )
    elif real_parent_tuple and real_max_steps:
        real_reports = _train_phase(
            trainer,
            "real_parent_finetuning",
            iter_parented_synthetic_training_batches,
            None,
            real_max_steps,
            target_loss,
            target_patience,
            real_parents=real_parent_tuple,
            synthetic_cache_dir=None,
            synthetic_cache_size=synthetic_cache_size,
            minimum_examples=None,
        )
    else:
        real_reports = PhaseReport(
            phase="real_parent_finetuning",
            reports=(),
            stopped_reason="no real parent exposures supplied" if not real_parent_tuple else "real_max_steps=0",
        )
    real_checkpoint = (
        _save_checkpoint(
            trainer,
            output_dir,
            "real_finetuned.pt",
            storage_cap_bytes,
            reserved_storage_bytes=synthetic_cache_bytes,
        )
        if real_phase_gate.is_open and real_parent_tuple and real_max_steps
        else None
    )

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
        "synthetic_examples_seen": real_phase_gate.synthetic_examples_seen,
        "synthetic_min_examples": real_phase_gate.minimum_synthetic_examples,
        "real_phase_gate": real_phase_gate,
        "synthetic_cache_path": str(synthetic_cache_path) if synthetic_cache_path else None,
        "synthetic_cache_bytes": synthetic_cache_bytes,
        "synthetic_cache_size_mib": synthetic_cache_size,
        "synthetic_cache_integration_note": (
            None
            if synthetic_cache_stream_supported
            else "pending stream support for cache_dir and cache_size"
        ),
        "real": real_reports,
        "held_out_real_parent_evaluation": held_out_eval,
        "synthetic_checkpoint": synthetic_checkpoint,
        "real_checkpoint": real_checkpoint,
        "rss_cap_bytes": rss_cap_bytes,
        "storage_cap_bytes": storage_cap_bytes,
    }


def _synthetic_curriculum_gate(
    synthetic_examples_seen: int,
    minimum_synthetic_examples: int,
) -> CurriculumGateReport:
    """Return whether the lazy synthetic stream has earned real-data admission."""
    is_open = synthetic_examples_seen >= minimum_synthetic_examples
    reason = (
        "synthetic warm-up complete"
        if is_open
        else (
            "synthetic warm-up incomplete: "
            f"{synthetic_examples_seen}/{minimum_synthetic_examples} synthetic examples seen"
        )
    )
    return CurriculumGateReport(
        synthetic_examples_seen=synthetic_examples_seen,
        minimum_synthetic_examples=minimum_synthetic_examples,
        is_open=is_open,
        reason=reason,
    )


def _prepare_synthetic_cache_dir(cache_dir: str | Path | None) -> Path | None:
    """Resolve an explicitly SSD-backed cache path without filling memory."""
    return Path(cache_dir).resolve() if cache_dir is not None else None


def _directory_bytes(directory: Path | None) -> int:
    if directory is None or not directory.exists():
        return 0
    return sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())


def _stream_supports_synthetic_cache(stream_factory: Any) -> bool:
    parameters = inspect.signature(stream_factory).parameters
    return "cache_dir" in parameters and "cache_size" in parameters


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
    synthetic_cache_dir: Path | None,
    synthetic_cache_size: int,
    minimum_examples: int | None = None,
) -> PhaseReport:
    reports: list[EpochReport] = []
    consecutive = 0
    for step in range(max_steps):
        if real_parents is None:
            stream_kwargs: dict[str, Any] = {
                "sample_count": 1,
                # Keep the counterfactual pair on CPU.  The runner splits it
                # before BoundedTrainer moves a single view to CUDA; retaining
                # a full paired GPU batch would defeat the memory bound.
                "device": "cpu",
            }
            if _stream_supports_synthetic_cache(stream_factory):
                stream_kwargs.update(
                    cache_dir=synthetic_cache_dir,
                    cache_size=synthetic_cache_size,
                )
            stream: Iterator[AstroMambaHTrainingBatch] = stream_factory(
                config,
                **stream_kwargs,
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
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "step": step,
                        "samples_seen": trainer.state.samples_seen,
                        "optimizer_steps": trainer.state.optimizer_steps,
                        "loss": report.last_loss,
                        "rss_within_cap": report.rss_within_cap,
                        "peak_gpu_mb": round((report.peak_gpu_memory_bytes or 0) / 1e6, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if report.last_loss is not None and report.last_loss <= target_loss:
                consecutive += 1
                if (
                    consecutive >= target_patience
                    and (minimum_examples is None or trainer.state.samples_seen >= minimum_examples)
                ):
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
    *,
    reserved_storage_bytes: int = 0,
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
    if total + reserved_storage_bytes > storage_cap_bytes:
        raise RuntimeError(
            "checkpoint storage cap exceeded: "
            f"{total + reserved_storage_bytes} > {storage_cap_bytes}"
        )
    return str(path)
