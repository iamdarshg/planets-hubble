import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


_SPEC = importlib.util.spec_from_file_location(
    "train_isolated_gpu", Path(__file__).parents[2] / "examples" / "train_isolated_gpu.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
counterfactual_checkpoint_paths = _MODULE.counterfactual_checkpoint_paths
synthetic_worker_shape_args = _MODULE.synthetic_worker_shape_args
_WORKER_SPEC = importlib.util.spec_from_file_location(
    "isolated_gpu_step", Path(__file__).parents[2] / "examples" / "isolated_gpu_step.py"
)
assert _WORKER_SPEC is not None and _WORKER_SPEC.loader is not None
_WORKER_MODULE = importlib.util.module_from_spec(_WORKER_SPEC)
_WORKER_SPEC.loader.exec_module(_WORKER_MODULE)
prepare_worker_batch = _WORKER_MODULE.prepare_worker_batch


def _optional(module, name):
    return getattr(module, name, None)


def test_counterfactual_checkpoint_paths_are_distinct_and_pair_scoped() -> None:
    positive, negative = counterfactual_checkpoint_paths(Path("run.pt"), 7)

    assert positive == Path("run.pair7.positive.pt")
    assert negative == Path("run.pair7.negative.pt")
    assert positive != negative


def test_synthetic_worker_shape_args_request_temporal_context() -> None:
    assert synthetic_worker_shape_args(1, 4, skip_dense_heatmaps=True) == [
        "--visits", "1", "--local-steps", "4", "--skip-dense-heatmaps"
    ]


def test_prepare_worker_batch_moves_before_training() -> None:
    class Movable:
        def __init__(self) -> None:
            self.device = None

        def to(self, device):
            self.device = device
            return self

    batch = Movable()
    assert prepare_worker_batch(batch, "cpu") is batch
    assert batch.device == "cpu"


def test_worker_command_propagates_auto_device_temporal_shape_and_view(tmp_path: Path) -> None:
    builder = _optional(_MODULE, "build_worker_command")
    assert callable(builder), "the orchestration boundary must build worker commands"

    command = builder(
        worker=Path("worker.py"),
        checkpoint=tmp_path / "negative.pt",
        seed=11,
        view=0,
        learning_rate=0.02,
        visits=3,
        local_steps=5,
        device="auto",
        skip_dense_heatmaps=True,
    )

    assert command == [
        sys.executable,
        "worker.py",
        "--checkpoint", str(tmp_path / "negative.pt"),
        "--seed", "11",
        "--view", "0",
        "--learning-rate", "0.02",
        "--device", "auto",
        "--visits", "3",
        "--local-steps", "5",
        "--skip-dense-heatmaps",
    ]


def test_paired_view_zero_is_null_and_uses_negative_checkpoint(tmp_path: Path) -> None:
    positive, negative = counterfactual_checkpoint_paths(tmp_path / "run.pt", 2)
    selector = _optional(_MODULE, "checkpoint_for_view")
    label_for_view = _optional(_MODULE, "label_for_view")
    assert callable(selector), "paired orchestration must select checkpoints by view"
    assert callable(label_for_view), "worker views must expose their semantic labels"

    assert selector(positive, negative, 0) == negative
    assert selector(positive, negative, 1) == positive
    assert label_for_view(0) == "null"
    assert label_for_view(1) == "injected"


def test_paired_workers_start_from_one_snapshot_and_cleanup_after_average_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "run.pt"
    checkpoint.write_bytes(b"same-initial-state")
    calls: list[list[str]] = []

    def fake_run(command, *, check, capture_output, text):
        del check, capture_output, text
        calls.append(command)
        if "--view" in command:
            worker_checkpoint = Path(command[command.index("--checkpoint") + 1])
            assert worker_checkpoint.read_bytes() == b"same-initial-state"
            view = command[command.index("--view") + 1]
            worker_checkpoint.write_bytes(f"updated-{view}".encode())
            return subprocess.CompletedProcess(command, 0, json.dumps({"view": int(view)}) + "\n", "")

        output = Path(command[command.index("--output") + 1])
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_bytes(b"failed-average-temp")
        return subprocess.CompletedProcess(command, 9, "", "average failed")

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_isolated_gpu.py",
            "--steps", "1",
            "--paired",
            "--checkpoint", str(checkpoint),
            "--visits", "2",
            "--local-steps", "4",
            "--device", "auto",
        ],
    )

    assert _MODULE.main() == 9
    capsys.readouterr()

    worker_calls = [call for call in calls if "--view" in call]
    assert len(worker_calls) == 2
    worker_paths = [Path(call[call.index("--checkpoint") + 1]) for call in worker_calls]
    assert worker_paths[0].name == "run.pair0.negative.pt"
    assert worker_paths[1].name == "run.pair0.positive.pt"
    assert worker_paths[0] != worker_paths[1]
    assert "--device" in worker_calls[0]
    assert worker_calls[0][worker_calls[0].index("--device") + 1] == "auto"
    assert worker_calls[0][worker_calls[0].index("--visits") + 1] == "2"
    assert worker_calls[0][worker_calls[0].index("--local-steps") + 1] == "4"
    assert not worker_paths[0].exists()
    assert not worker_paths[1].exists()
    assert not checkpoint.with_suffix(checkpoint.suffix + ".tmp").exists()


def test_worker_atomic_checkpoint_save_removes_temporary_file_on_save_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_checkpoint = _optional(_WORKER_MODULE, "save_checkpoint_atomically")
    assert callable(save_checkpoint), "worker checkpoint writes must have a cleanup boundary"
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"old-checkpoint")

    def fail_save(*args, **kwargs):
        del args, kwargs
        raise OSError("simulated checkpoint write failure")

    monkeypatch.setattr(_WORKER_MODULE.torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated checkpoint write failure"):
        save_checkpoint(checkpoint, {"model": {}})

    assert checkpoint.read_bytes() == b"old-checkpoint"
    assert not checkpoint.with_suffix(checkpoint.suffix + ".tmp").exists()


def test_worker_report_includes_measured_rss_and_storage_caps() -> None:
    payload_builder = _optional(_WORKER_MODULE, "worker_report_payload")
    assert callable(payload_builder), "worker output must report resource-cap evidence"
    report = SimpleNamespace(
        last_loss=0.25,
        loss_is_finite=True,
        process_rss_bytes=1234,
        rss_cap_bytes=2000,
        rss_within_cap=True,
        peak_gpu_memory_bytes=4567,
        storage_bytes_written=0,
        storage_cap_bytes=5000,
        storage_within_cap=True,
        resource_cap_violations=(),
    )

    payload = payload_builder(report, seed=8, view=0, checkpoint_bytes=321)

    assert payload["label"] == "null"
    assert payload["process_rss_bytes"] == 1234
    assert payload["rss_cap_bytes"] == 2000
    assert payload["rss_within_cap"] is True
    assert payload["peak_gpu_memory_bytes"] == 4567
    assert payload["storage_bytes_written"] == 321
    assert payload["storage_cap_bytes"] == 5000
    assert payload["storage_within_cap"] is True
    assert payload["resource_cap_violations"] == []


def test_worker_auto_device_resolution_uses_training_device_policy() -> None:
    resolver = _optional(_WORKER_MODULE, "resolve_worker_device")
    assert callable(resolver), "worker must resolve the auto device through the training policy"
    resolved = resolver("auto")
    assert resolved.type in {"cpu", "cuda"}
