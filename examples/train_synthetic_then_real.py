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
    parser.add_argument("--real-steps", type=int, default=16)
    parser.add_argument("--target-loss", type=float, default=0.05)
    parser.add_argument("--target-patience", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    construction_context = torch.device(device) if device.type == "cuda" else nullcontext()
    with construction_context:
        model = AstroMambaHTrainingAdapter(config=research_config())
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
        bounded_smoke_test=args.bounded_smoke_test,
        synthetic_cache_dir=args.synthetic_cache_dir,
        synthetic_cache_size=args.synthetic_cache_size,
        real_max_steps=args.real_steps,
        target_loss=args.target_loss,
        target_patience=args.target_patience,
        output_dir=args.output_dir,
    )
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
