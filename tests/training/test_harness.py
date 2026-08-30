import pytest
import torch
from torch import nn

from training import (
    BoundedTrainer,
    DEFAULT_RSS_CAP_BYTES,
    NonFiniteTrainingError,
    TinyAstroAdapter,
    TrainingConfig,
    make_tiny_adapter_batch,
    resolve_device,
)
from training.adapters import AstroMambaHTrainingBatch


def test_cpu_training_is_bounded_and_reports_finite_state():
    torch.manual_seed(7)
    model = TinyAstroAdapter()
    config = TrainingConfig(device="cpu", max_batches_per_epoch=2, amp="auto")
    trainer = BoundedTrainer(model, config=config)
    batches = [make_tiny_adapter_batch(batch_size=2) for _ in range(5)]

    report = trainer.train_epoch(batches)

    assert report.batches_seen == 2
    assert report.samples_seen == 4
    assert report.optimizer_steps == 2
    assert report.device == "cpu"
    assert report.amp_enabled is False
    assert report.loss_is_finite is True
    assert report.last_loss is not None
    assert torch.isfinite(torch.tensor(report.last_loss))
    assert DEFAULT_RSS_CAP_BYTES == int(1.8 * 1024 * 1024 * 1024)
    assert report.rss_cap_bytes == DEFAULT_RSS_CAP_BYTES
    assert report.storage_cap_bytes == 5 * 1024 * 1024 * 1024
    assert report.storage_bytes_written == 0
    assert report.storage_within_cap is True


def test_device_selection_rejects_unavailable_explicit_cuda():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda"}

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
            resolve_device("cuda")


def test_cpu_amp_is_disabled_and_checkpoint_report_has_no_tensor_payload():
    model = TinyAstroAdapter()
    trainer = BoundedTrainer(model, config=TrainingConfig(device="cpu", amp=True))

    assert trainer.amp_enabled is False

    report = trainer.checkpoint_report()

    assert report.model_parameter_count == sum(p.numel() for p in model.parameters())
    assert report.model_state_tensor_count > 0
    assert report.optimizer_state_tensor_count == 0
    assert "model_state_dict" not in report.__dict__
    assert report.storage_bytes_written == 0


def test_nonfinite_loss_fails_before_optimizer_step():
    model = TinyAstroAdapter()
    trainer = BoundedTrainer(model, config=TrainingConfig(device="cpu", max_batches_per_epoch=1))

    def nonfinite_loss(_prediction, _batch):
        return torch.tensor(float("nan"), requires_grad=True)

    with pytest.raises(NonFiniteTrainingError, match="loss is not finite"):
        trainer.train_epoch(
            [make_tiny_adapter_batch(batch_size=1)],
            loss_fn=nonfinite_loss,
        )

    assert trainer.state.optimizer_steps == 0


def test_gradient_nonfinite_check_fails_before_optimizer_step():
    class NaNGradientModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, _batch):
            return self.weight * 1.0

    model = NaNGradientModel()
    trainer = BoundedTrainer(model, config=TrainingConfig(device="cpu", max_batches_per_epoch=1))

    def nan_gradient_loss(prediction, _batch):
        prediction.register_hook(
            lambda gradient: gradient * torch.tensor(float("nan"), device=gradient.device)
        )
        return prediction.square()

    with pytest.raises(NonFiniteTrainingError, match="gradient is not finite"):
        trainer.train_epoch(
            [make_tiny_adapter_batch(batch_size=1)],
            loss_fn=nan_gradient_loss,
        )

    assert trainer.state.optimizer_steps == 0


def test_default_loss_uses_global_and_auxiliary_logits_with_multi_visit_shapes():
    prediction = {
        "head_logits": {"event": torch.tensor([0.2, -0.3], requires_grad=True)},
        "visit_event_logits": torch.tensor([[0.1, -0.2], [0.3, 0.4]], requires_grad=True),
    }
    batch = AstroMambaHTrainingBatch(
        inputs=make_tiny_adapter_batch(batch_size=2),
        target=torch.tensor([[1.0], [0.0]]),
        auxiliary_targets={"visit_event": torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
    )

    loss = __import__("training").default_loss_fn(prediction, batch)
    assert loss.ndim == 0
    loss.backward()
    assert prediction["head_logits"]["event"].grad is not None
    assert prediction["visit_event_logits"].grad is not None
