"""Run the bounded synthetic-first training lifecycle.

This entry point requires real parents only for the optional fine-tuning phase;
without them it performs the synthetic phase and records that the real phase
was intentionally skipped.  A future data-ingestion command can construct
``RealObservationParent`` values with ``ManifestParentLoader`` and pass them to
the same runner.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import replace
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model import research_config  # noqa: E402
from synthetic import SyntheticConfig  # noqa: E402
from training import (  # noqa: E402
    DEFAULT_SYNTHETIC_CACHE_DIR,
    DEFAULT_SYNTHETIC_CACHE_SIZE_MIB,
    DEFAULT_SYNTHETIC_MIN_EXAMPLES,
    AstroMambaHTrainingAdapter,
    resolve_device,
    train_synthetic_then_real,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--synthetic-steps", type=int, default=DEFAULT_SYNTHETIC_MIN_EXAMPLES // 2)
    parser.add_argument("--synthetic-min-examples", type=int, default=DEFAULT_SYNTHETIC_MIN_EXAMPLES)
    parser.add_argument("--synthetic-start-index", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--bf16-weights",
        action="store_true",
        help="keep trainable weights in bfloat16 to reduce host/GPU memory; "
        "use a larger learning rate (e.g. 1e-2) so updates stay representable",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="optional checkpoint to resume from; enables chunked long training "
        "with fresh processes so the Windows/CUDA workspace growth never stalls a run",
    )
    parser.add_argument(
        "--synthetic-cache-dir",
        type=Path,
        default=DEFAULT_SYNTHETIC_CACHE_DIR,
        help="SSD-backed directory for the bounded procedural cache",
    )
    parser.add_argument(
        "--synthetic-cache-size",
        type=int,
        default=DEFAULT_SYNTHETIC_CACHE_SIZE_MIB,
        help="procedural cache budget in MiB",
    )
    parser.add_argument(
        "--bounded-smoke-test",
        action="store_true",
        help="allow a smaller synthetic warm-up only for a bounded smoke test",
    )
    parser.add_argument(
        "--decode-heatmaps",
        action="store_true",
        help="enable the dense wavelength heatmap decoder during training. "
        "The default keeps training cap-safe under the 1.6 GiB host-RSS cap; "
        "the decoder accounts for roughly 13M of the 82.5M parameters.",
    )
    parser.add_argument("--real-steps", type=int, default=16)
    parser.add_argument("--target-loss", type=float, default=0.05)
    parser.add_argument("--target-patience", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    construction_context = torch.device(device) if device.type == "cuda" else nullcontext()
    model_config = research_config()
    if not args.decode_heatmaps:
        model_config = replace(model_config, decode_heatmaps=False)
    with construction_context:
        model = AstroMambaHTrainingAdapter(config=model_config)
    if args.resume_from is not None:
        if not args.resume_from.is_file():
            raise FileNotFoundError(args.resume_from)
        state = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    if args.bf16_weights:
        model = model.to(device, dtype=torch.bfloat16)
    result = train_synthetic_then_real(
        model=model,
        synthetic_config=SyntheticConfig(
            seed=23,
            visits=1,
            local_steps=1,
            raster_height=720,
            raster_width=1280,
            wavelength_nm=(450.0, 650.0, 1000.0),
        ),
        device=device,
        synthetic_max_steps=args.synthetic_steps,
        synthetic_min_examples=args.synthetic_min_examples,
        synthetic_start_index=args.synthetic_start_index,
        bounded_smoke_test=args.bounded_smoke_test,
        synthetic_cache_dir=args.synthetic_cache_dir,
        synthetic_cache_size=args.synthetic_cache_size,
        optimizer_learning_rate=args.learning_rate,
        real_max_steps=args.real_steps,
        target_loss=args.target_loss,
        target_patience=args.target_patience,
        output_dir=args.output_dir,
    )
    result["model_parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps(_jsonable(result), sort_keys=True))
    return 0


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
