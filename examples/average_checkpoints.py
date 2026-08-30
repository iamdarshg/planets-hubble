"""Average synchronized counterfactual checkpoints on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--negative", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    positive = torch.load(args.positive, map_location="cpu", weights_only=False)
    negative = torch.load(args.negative, map_location="cpu", weights_only=False)
    positive_state = positive["model"] if "model" in positive else positive
    negative_state = negative["model"] if "model" in negative else negative
    if positive_state.keys() != negative_state.keys():
        raise ValueError("counterfactual checkpoints do not have matching state keys")
    averaged = {}
    for name, value in positive_state.items():
        other = negative_state[name]
        if not isinstance(value, torch.Tensor) or not isinstance(other, torch.Tensor):
            averaged[name] = value
        elif value.is_floating_point():
            averaged[name] = ((value.float() + other.float()) * 0.5).to(value.dtype)
        else:
            averaged[name] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(
        {
            "model": averaged,
            "parameter_count": positive.get("parameter_count") if isinstance(positive, dict) else None,
            "counterfactual_update": "synchronized_positive_negative_weight_average",
        },
        temporary,
    )
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "parameters": len(averaged)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
