"""Evaluate one synthetic counterfactual pair with a saved checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import research_config  # noqa: E402
from synthetic import SyntheticConfig  # noqa: E402
from training import AstroMambaHTrainingAdapter, iter_paired_synthetic_training_batches, resolve_device  # noqa: E402
from training.pipeline import _split_batch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-dense-heatmaps",
        action="store_true",
        help="match the cap-safe decoder-off training configuration",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    config = research_config()
    if args.skip_dense_heatmaps:
        config = replace(config, decode_heatmaps=False)
    with (torch.device(device) if device.type == "cuda" else nullcontext()):
        model = AstroMambaHTrainingAdapter(config=config)
    model = model.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    pair = next(
        iter_paired_synthetic_training_batches(
            SyntheticConfig(
                seed=args.seed,
                visits=1,
                local_steps=1,
                raster_height=720,
                raster_width=1280,
                wavelength_nm=(450.0, 650.0, 1000.0),
            ),
            sample_count=1,
            device="cpu",
        )
    )
    model.eval()
    rows = []
    with torch.inference_mode():
        for view, batch in enumerate(_split_batch(pair)):
            batch = batch.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                enabled=device.type == "cuda",
            ):
                output = model(batch)
            logit = output["global_event_logits"].reshape(-1)[0]
            direct_logit = output["source_photometry_event_logits"].reshape(-1)[0]
            source_logits = output["source_logits"]
            rows.append(
                {
                    "view": view,
                    "target": float(batch.target.reshape(-1)[0].cpu()),
                    "global_event_logit": float(logit.float().cpu()),
                    "global_event_probability": float(logit.float().sigmoid().cpu()),
                    "source_photometry_event_logit": float(direct_logit.float().cpu()),
                    "source_logit_min": float(source_logits.float().min().cpu()),
                    "source_logit_max": float(source_logits.float().max().cpu()),
                }
            )
    print(json.dumps({"checkpoint": str(args.checkpoint), "seed": args.seed, "views": rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
