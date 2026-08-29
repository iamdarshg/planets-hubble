"""One-batch CUDA smoke test for the local training harness."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from typing import Optional

import torch

from .adapters import AstroMambaHTrainingAdapter, make_tiny_astromamba_batch
from .harness import (
    DEFAULT_RSS_CAP_BYTES,
    DEFAULT_STORAGE_CAP_BYTES,
    BoundedTrainer,
    TrainingConfig,
)


@dataclass(frozen=True)
class GpuSmokeResult:
    skipped: bool
    reason: Optional[str]
    device: str
    batches_seen: int
    loss_is_finite: bool
    amp_enabled: bool
    amp_dtype: Optional[str]
    model_name: str
    cuda_runtime_version: Optional[str]
    gpu_name: Optional[str]
    input_raster_shape: tuple[int, ...]
    rss_cap_bytes: int
    rss_within_cap: Optional[bool]
    storage_cap_bytes: int
    storage_bytes_written: int
    storage_within_cap: bool
    resource_cap_violations: tuple[str, ...]
    peak_gpu_memory_bytes: Optional[int]
    process_rss_bytes: Optional[int]


def run_gpu_smoke() -> GpuSmokeResult:
    """Run one compact CUDA optimizer step, or return a clear skip result."""

    if not torch.cuda.is_available():
        return GpuSmokeResult(
            skipped=True,
            reason="CUDA unavailable: local GPU smoke test not run",
            device="cpu",
            batches_seen=0,
            loss_is_finite=False,
            amp_enabled=False,
            amp_dtype=None,
            model_name="none",
            cuda_runtime_version=None,
            gpu_name=None,
            input_raster_shape=(),
            rss_cap_bytes=DEFAULT_RSS_CAP_BYTES,
            rss_within_cap=None,
            storage_cap_bytes=DEFAULT_STORAGE_CAP_BYTES,
            storage_bytes_written=0,
            storage_within_cap=True,
            resource_cap_violations=(),
            peak_gpu_memory_bytes=None,
            process_rss_bytes=None,
        )

    device = torch.device("cuda")
    model = AstroMambaHTrainingAdapter().to(device)
    batch = make_tiny_astromamba_batch(device=device)
    trainer = BoundedTrainer(
        model,
        config=TrainingConfig(device=device, max_batches_per_epoch=1, amp="auto"),
    )
    report = trainer.train_epoch([batch])
    return GpuSmokeResult(
        skipped=False,
        reason=None,
        device=report.device,
        batches_seen=report.batches_seen,
        loss_is_finite=report.loss_is_finite,
        amp_enabled=report.amp_enabled,
        amp_dtype=report.amp_dtype,
        model_name=model.model_name,
        cuda_runtime_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(device),
        input_raster_shape=tuple(batch.inputs.raster.shape),
        rss_cap_bytes=report.rss_cap_bytes,
        rss_within_cap=report.rss_within_cap,
        storage_cap_bytes=report.storage_cap_bytes,
        storage_bytes_written=report.storage_bytes_written,
        storage_within_cap=report.storage_within_cap,
        resource_cap_violations=report.resource_cap_violations,
        peak_gpu_memory_bytes=report.peak_gpu_memory_bytes,
        process_rss_bytes=report.process_rss_bytes,
    )


def main() -> int:
    print(json.dumps(asdict(run_gpu_smoke()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
