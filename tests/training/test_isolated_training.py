import importlib.util
from pathlib import Path


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
