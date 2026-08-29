"""Bounded, device-aware local training utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DeviceRequest = str | torch.device
AmpRequest = Literal["auto"] | bool
LossFunction = Callable[[Any, Any], Tensor]
DEFAULT_RSS_CAP_BYTES = 720 * 1024 * 1024
DEFAULT_STORAGE_CAP_BYTES = 5 * 1024 * 1024 * 1024


class NonFiniteTrainingError(RuntimeError):
    """Raised when loss or gradients are NaN/Inf before an optimizer step."""


def resolve_device(requested: DeviceRequest = "auto") -> torch.device:
    """Resolve a requested device without silently falling back from CUDA."""

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable")
        return torch.device("cuda", index)
    if device.type == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"unsupported training device: {device}")


def amp_supported(device: torch.device) -> bool:
    """Return whether this harness has a supported AMP path for ``device``."""

    return bool(
        device.type == "cuda"
        and torch.cuda.is_available()
        and hasattr(torch, "amp")
        and hasattr(torch.amp, "autocast")
        and hasattr(torch.amp, "GradScaler")
    )


def process_rss_bytes() -> Optional[int]:
    """Return process RSS when available, without adding a hard dependency."""

    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


@dataclass(frozen=True)
class TrainingConfig:
    """Explicit limits and optimization settings for a local run."""

    device: DeviceRequest = "auto"
    max_batches_per_epoch: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    amp: AmpRequest = "auto"
    grad_clip_norm: Optional[float] = None
    rss_cap_bytes: int = DEFAULT_RSS_CAP_BYTES
    storage_cap_bytes: int = DEFAULT_STORAGE_CAP_BYTES

    def __post_init__(self) -> None:
        if self.max_batches_per_epoch < 1:
            raise ValueError("max_batches_per_epoch must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive when provided")
        if self.rss_cap_bytes < 1 or self.storage_cap_bytes < 1:
            raise ValueError("resource caps must be positive")
        if self.amp not in ("auto", True, False):
            raise ValueError("amp must be 'auto', True, or False")


@dataclass
class TrainingState:
    epoch: int = 0
    global_step: int = 0
    batches_seen: int = 0
    samples_seen: int = 0
    optimizer_steps: int = 0
    last_loss: Optional[float] = None
    loss_is_finite: bool = True


@dataclass(frozen=True)
class EpochReport:
    epoch: int
    batches_seen: int
    samples_seen: int
    optimizer_steps: int
    last_loss: Optional[float]
    loss_is_finite: bool
    device: str
    amp_enabled: bool
    amp_dtype: Optional[str]
    parameter_count: int
    rss_cap_bytes: int
    rss_within_cap: Optional[bool]
    storage_cap_bytes: int
    storage_bytes_written: int
    storage_within_cap: bool
    resource_cap_violations: tuple[str, ...]
    peak_gpu_memory_bytes: Optional[int] = None
    process_rss_bytes: Optional[int] = None


@dataclass(frozen=True)
class CheckpointReport:
    """Metadata sufficient to audit a checkpoint without storing its tensors."""

    epoch: int
    global_step: int
    model_parameter_count: int
    model_state_tensor_count: int
    model_state_bytes: int
    optimizer_state_tensor_count: int
    optimizer_state_bytes: int
    device: str
    amp_enabled: bool
    amp_dtype: Optional[str]
    storage_bytes_written: int


def _batch_size(batch: Any) -> int:
    if hasattr(batch, "batch_size"):
        return int(batch.batch_size)
    if isinstance(batch, Mapping):
        for name in ("target", "labels", "features"):
            value = batch.get(name)
            if isinstance(value, Tensor) and value.ndim > 0:
                return int(value.shape[0])
    if isinstance(batch, Tensor) and batch.ndim > 0:
        return int(batch.shape[0])
    if hasattr(batch, "inputs") and hasattr(batch.inputs, "raster"):
        return int(batch.inputs.raster.shape[0])
    raise TypeError("batch must expose batch_size or a batched tensor field")


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if hasattr(value, "to") and callable(value.to):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _target_from_batch(batch: Any) -> Tensor:
    if hasattr(batch, "target"):
        return batch.target
    if isinstance(batch, Mapping) and isinstance(batch.get("target"), Tensor):
        return batch["target"]
    raise TypeError("default_loss requires batch.target or batch['target']")


def default_loss_fn(prediction: Any, batch: Any) -> Tensor:
    """Binary event loss for tensor predictions or common model head outputs."""

    target = _target_from_batch(batch).to(dtype=torch.float32)
    if isinstance(prediction, Mapping):
        if "prediction" in prediction:
            prediction = prediction["prediction"]
        elif "logits" in prediction:
            prediction = prediction["logits"]
        else:
            heads = prediction.get("global_heads", prediction)
            prediction = heads["event_probability"]
            with torch.autocast(
                device_type=prediction.device.type,
                enabled=False,
            ):
                return F.binary_cross_entropy(prediction.float(), target)
    if not isinstance(prediction, Tensor):
        raise TypeError("model output must be a Tensor or supported mapping")
    return F.binary_cross_entropy_with_logits(prediction, target)


def _finite_gradients(model: nn.Module) -> bool:
    found_gradient = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found_gradient = True
        if not torch.isfinite(parameter.grad).all().item():
            return False
    if not found_gradient:
        raise RuntimeError("no parameter gradients were produced")
    return True


def _tensor_stats(state: Mapping[str, Any]) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    for value in state.values():
        if isinstance(value, Tensor):
            count += 1
            total_bytes += value.numel() * value.element_size()
        elif isinstance(value, Mapping):
            nested_count, nested_bytes = _tensor_stats(value)
            count += nested_count
            total_bytes += nested_bytes
    return count, total_bytes


class BoundedTrainer:
    """A small optimizer loop with explicit resource and numerical guards."""

    def __init__(
        self,
        model: nn.Module,
        *,
        config: TrainingConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.device = resolve_device(config.device)
        self.config = config
        self.model = model.to(self.device)
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("model must have at least one trainable parameter")
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        if config.amp == "auto":
            self.amp_enabled = amp_supported(self.device)
        else:
            self.amp_enabled = bool(config.amp and amp_supported(self.device))
        if self.amp_enabled and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif self.amp_enabled:
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = None
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
            if self.amp_enabled and self.amp_dtype == torch.float16
            else None
        )
        self.state = TrainingState()

    def _autocast(self):
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype or torch.float32,
            enabled=self.amp_enabled,
        )

    def train_epoch(
        self,
        batches: Iterable[Any],
        *,
        loss_fn: LossFunction = default_loss_fn,
    ) -> EpochReport:
        """Train on at most ``max_batches_per_epoch`` batches."""

        self.model.train()
        epoch_batches = 0
        epoch_samples = 0
        peak_gpu_memory = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        for batch in batches:
            if epoch_batches >= self.config.max_batches_per_epoch:
                break
            moved_batch = _move_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                prediction = self.model(moved_batch)
                loss = loss_fn(prediction, moved_batch)
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            if not torch.isfinite(loss).item():
                self.state.loss_is_finite = False
                raise NonFiniteTrainingError("loss is not finite")

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            if not _finite_gradients(self.model):
                raise NonFiniteTrainingError("gradient is not finite")
            if self.config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            epoch_batches += 1
            epoch_samples += _batch_size(moved_batch)
            self.state.global_step += 1
            self.state.batches_seen += 1
            self.state.samples_seen += _batch_size(moved_batch)
            self.state.optimizer_steps += 1
            self.state.last_loss = float(loss.detach().cpu())
            self.state.loss_is_finite = True

        self.state.epoch += 1
        if self.device.type == "cuda":
            peak_gpu_memory = int(torch.cuda.max_memory_allocated(self.device))
        rss = process_rss_bytes()
        rss_within_cap = None if rss is None else rss <= self.config.rss_cap_bytes
        storage_bytes_written = 0
        storage_within_cap = storage_bytes_written <= self.config.storage_cap_bytes
        violations = []
        if rss_within_cap is False:
            violations.append("rss")
        if not storage_within_cap:
            violations.append("storage")
        return EpochReport(
            epoch=self.state.epoch,
            batches_seen=epoch_batches,
            samples_seen=epoch_samples,
            optimizer_steps=self.state.optimizer_steps,
            last_loss=self.state.last_loss,
            loss_is_finite=self.state.loss_is_finite,
            device=str(self.device),
            amp_enabled=self.amp_enabled,
            amp_dtype=str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else None,
            parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            rss_cap_bytes=self.config.rss_cap_bytes,
            rss_within_cap=rss_within_cap,
            storage_cap_bytes=self.config.storage_cap_bytes,
            storage_bytes_written=storage_bytes_written,
            storage_within_cap=storage_within_cap,
            resource_cap_violations=tuple(violations),
            peak_gpu_memory_bytes=peak_gpu_memory,
            process_rss_bytes=rss,
        )

    def checkpoint_report(self) -> CheckpointReport:
        """Report checkpoint size/state metadata without serializing tensors."""

        model_count, model_bytes = _tensor_stats(self.model.state_dict())
        optimizer_count, optimizer_bytes = _tensor_stats(self.optimizer.state_dict())
        return CheckpointReport(
            epoch=self.state.epoch,
            global_step=self.state.global_step,
            model_parameter_count=sum(parameter.numel() for parameter in self.model.parameters()),
            model_state_tensor_count=model_count,
            model_state_bytes=model_bytes,
            optimizer_state_tensor_count=optimizer_count,
            optimizer_state_bytes=optimizer_bytes,
            device=str(self.device),
            amp_enabled=self.amp_enabled,
            amp_dtype=str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else None,
            storage_bytes_written=0,
        )
