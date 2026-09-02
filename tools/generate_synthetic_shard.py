"""Render one deterministic shard of full-resolution synthetic pairs.

The rendered arrays are intentionally released after their fingerprint is
recorded.  Persisting every 720x1280 pair would exceed the repository's
bounded storage contract; the sample index and generator seed are sufficient
to replay the exact arrays during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic import SyntheticConfig, SyntheticGenerator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--pair-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=64)
    parser.add_argument(
        "--fingerprint-raster",
        action="store_true",
        help="hash both full-resolution raster views; slower but stronger audit evidence",
    )
    args = parser.parse_args()
    if args.start_index < 0 or args.pair_count < 1:
        raise ValueError("start-index must be non-negative and pair-count must be positive")
    if args.progress_every < 1:
        raise ValueError("progress-every must be positive")

    config = SyntheticConfig(
        seed=args.seed,
        visits=1,
        local_steps=1,
        raster_height=720,
        raster_width=1280,
        wavelength_nm=(450.0, 650.0, 1000.0),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    temporary = args.output.with_name(f".{args.output.name}.{__import__('os').getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for local_index in range(args.pair_count):
                sample_index = args.start_index + local_index
                bundle = SyntheticGenerator(
                    replace(config, seed=config.seed + sample_index)
                ).generate()
                digest = hashlib.sha256() if args.fingerprint_raster else None
                for view_name in ("null", "injected"):
                    arrays = bundle.as_model_numpy(view_name)
                    if digest is not None:
                        raster = np.ascontiguousarray(arrays["raster"])
                        digest.update(raster.tobytes(order="C"))
                row = {
                    "pair_index": local_index,
                    "sample_index": sample_index,
                    "generator_seed": config.seed + sample_index,
                    "views": 2,
                    "targets": [
                        float(bundle.null.labels.latent_positive),
                        float(bundle.injected.labels.latent_positive),
                    ],
                    "raster_shape": list(bundle.as_model_numpy("null")["raster"].shape),
                }
                if digest is not None:
                    row["sha256_raster_pair"] = digest.hexdigest()
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                if (local_index + 1) % args.progress_every == 0:
                    handle.flush()
                    print(
                        json.dumps(
                            {
                                "output": str(args.output),
                                "pairs_generated": local_index + 1,
                                "views_generated": (local_index + 1) * 2,
                                "elapsed_seconds": round(time.time() - started, 1),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "pair_count": args.pair_count,
                "view_count": args.pair_count * 2,
                "sample_index_first": args.start_index,
                "sample_index_last": args.start_index + args.pair_count - 1,
                "generator_seed_first": config.seed + args.start_index,
                "generator_seed_last": config.seed + args.start_index + args.pair_count - 1,
                "elapsed_seconds": round(time.time() - started, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
