"""Evaluate a trained checkpoint on a held-out real HST parent sequence."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import research_config  # noqa: E402
from training import AstroMambaHTrainingAdapter, evaluate_parent_injections, resolve_device  # noqa: E402
from isolated_gpu_step import load_parent  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--exposure-index",
        type=int,
        default=None,
        help="optional single-exposure diagnostic; default evaluates the full parent sequence",
    )
    parser.add_argument(
        "--sequence-summary",
        action="store_true",
        help="reduce the complete parent sequence to the cap-safe temporal summary",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    with (torch.device(device) if device.type == "cuda" else nullcontext()):
        model = AstroMambaHTrainingAdapter(config=research_config())
    model = model.to(device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    report = evaluate_parent_injections(
        model,
        (load_parent(args.manifest, exposure_index=args.exposure_index),),
        device=device,
        sample_count=2,
        sequence_summary=args.sequence_summary,
    )
    print(json.dumps(dataclasses.asdict(report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
