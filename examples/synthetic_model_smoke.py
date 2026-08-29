"""Generate one synthetic 720p bundle and train AstroMamba-H for one step.

This is intentionally a bounded integration smoke test.  It exercises the
synthetic generator, normalized model-input contract, and generic trainer in
one process; it does not claim scientific convergence.

Run from the repository root with::

    python examples/synthetic_model_smoke.py --device cuda
    python examples/synthetic_model_smoke.py --device cuda --research
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import AstroMambaHInputs, research_config  # noqa: E402
from synthetic import SyntheticConfig, SyntheticGenerator  # noqa: E402
from training import (  # noqa: E402
    AstroMambaHTrainingAdapter,
    AstroMambaHTrainingBatch,
    BoundedTrainer,
    TrainingConfig,
    resolve_device,
    tiny_astromamba_config,
)
def make_synthetic_training_batch(device: torch.device) -> AstroMambaHTrainingBatch:
    """Create one full-raster normalized bundle without writing an artifact."""

    bundle = SyntheticGenerator(
        SyntheticConfig(
            seed=23,
            visits=1,
            local_steps=1,
            raster_height=720,
            raster_width=1280,
            wavelength_nm=(450.0, 650.0, 1000.0),
        )
    ).generate()
    arrays = bundle.as_model_numpy("injected")
    inputs = AstroMambaHInputs(
        **{
            name: torch.from_numpy(value).to(device)
            for name, value in arrays.items()
        }
    )
    target = torch.tensor(
        [[1.0 if bundle.labels is not None and bundle.labels.latent_positive else 0.0]],
        dtype=torch.float32,
        device=device,
    )
    return AstroMambaHTrainingBatch(inputs=inputs, target=target)


def run(device_request: str = "auto", *, research: bool = False) -> dict[str, object]:
    device = resolve_device(device_request)
    config = research_config() if research else tiny_astromamba_config()
    batch = make_synthetic_training_batch(device)
    model = AstroMambaHTrainingAdapter(config=config)
    trainer = BoundedTrainer(
        model,
        config=TrainingConfig(device=device, max_batches_per_epoch=1, amp="auto"),
    )
    report = trainer.train_epoch([batch])
    return {
        "device": report.device,
        "model_name": model.model_name,
        "parameter_count": report.parameter_count,
        "configuration": "research" if research else "tiny",
        "input_raster_shape": list(batch.inputs.raster.shape),
        "batches_seen": report.batches_seen,
        "loss_is_finite": report.loss_is_finite,
        "last_loss": report.last_loss,
        "amp_enabled": report.amp_enabled,
        "amp_dtype": report.amp_dtype,
        "peak_gpu_memory_bytes": report.peak_gpu_memory_bytes,
        "process_rss_bytes": report.process_rss_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--research",
        action="store_true",
        help="run the measured 84M-parameter research configuration",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.device, research=args.research), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
