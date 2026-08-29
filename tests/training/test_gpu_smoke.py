import pytest
import torch

from training import DEFAULT_RSS_CAP_BYTES
from training.gpu_smoke import run_gpu_smoke


def test_gpu_smoke_runs_or_skips_with_explicit_reason():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: local GPU smoke test not run")

    result = run_gpu_smoke()

    assert result.skipped is False
    assert result.device.startswith("cuda")
    assert result.batches_seen == 1
    assert result.loss_is_finite is True
    assert result.amp_enabled is True
    assert result.model_name == "AstroMambaH"
    assert result.cuda_runtime_version is not None
    assert result.gpu_name is not None
    assert result.input_raster_shape == (1, 1, 1, 6, 720, 1280)
    assert result.peak_gpu_memory_bytes is not None
    assert result.rss_cap_bytes == DEFAULT_RSS_CAP_BYTES
    assert result.storage_cap_bytes == 5 * 1024 * 1024 * 1024
    assert result.storage_bytes_written == 0
    assert result.storage_within_cap is True
    assert result.rss_within_cap == (
        result.process_rss_bytes is not None
        and result.process_rss_bytes <= result.rss_cap_bytes
    )
    assert ("rss" in result.resource_cap_violations) == (not result.rss_within_cap)
