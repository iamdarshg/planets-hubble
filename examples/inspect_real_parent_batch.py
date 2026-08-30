"""Inspect positive/null parent batches and their model logits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import research_config  # noqa: E402
from training import AstroMambaHTrainingAdapter, iter_parented_synthetic_training_batches  # noqa: E402
from isolated_gpu_step import _cast_batch_floating_tensors, load_parent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-summary", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda", torch.cuda.current_device())
    model = AstroMambaHTrainingAdapter(config=research_config()).to(device, dtype=torch.bfloat16)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    parent = load_parent(args.manifest)
    rows = []
    with torch.inference_mode():
        for batch in iter_parented_synthetic_training_batches(
            (parent,), sample_count=2, device="cpu", sequence_summary=args.sequence_summary
        ):
            raster = batch.inputs.raster
            batch = _cast_batch_floating_tensors(batch, torch.bfloat16).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
            rows.append(
                {
                    "target": float(batch.target.item()),
                    "raster_min": float(raster.min()),
                    "raster_max": float(raster.max()),
                    "global_event_logit": float(output["global_event_logits"].float().item()),
                    "global_event_probability": float(output["global_event_logits"].float().sigmoid().item()),
                    "head_event_logit": float(output["head_logits"]["event"].float().reshape(-1)[0].item()),
                    "visit_event_logit": float(output["visit_event_logits"].float().reshape(-1)[0].item()),
                    "source_event_logit": float(output["source_event_logits"].float().reshape(-1)[0].item()),
                }
            )
    print(json.dumps({"rows": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
