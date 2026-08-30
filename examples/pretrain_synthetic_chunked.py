"""Chunked synthetic pretraining driver for long, cap-safe runs.

The Windows/CUDA build used for this project accumulates GPU workspace state
across repeated in-process research-model steps and eventually stalls.  This
driver runs the two-phase entry point in short chunks, each in a fresh process,
resuming from the previous checkpoint via --resume-from and advancing the
procedural sample counter with --synthetic-start-index.  Chunks that are killed
by the resource watchdog are retried (bounded) after a settle delay.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=2048)
    parser.add_argument("--chunk-steps", type=int, default=40)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-loss", type=float, default=0.05)
    parser.add_argument("--target-patience", type=int, default=3)
    parser.add_argument(
        "--bounded-smoke-test",
        action="store_true",
        help="allow chunks with fewer than the production warm-up (debug only)",
    )
    parser.add_argument("--max-chunk-retries", type=int, default=3)
    parser.add_argument("--settle-seconds", type=int, default=20)
    parser.add_argument(
        "--bf16-weights",
        action="store_true",
        help="pass --bf16-weights and --learning-rate 1e-2 to each chunk "
        "process to keep WorkingSet under the 1.6 GiB cap",
    )
    args = parser.parse_args()
    if args.total_steps < 1 or args.chunk_steps < 1:
        raise ValueError("total-steps and chunk-steps must be positive")
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.max_chunk_retries < 1 or args.settle_seconds < 0:
        raise ValueError("retries must be positive and settle-seconds non-negative")

    entry = Path(__file__).resolve().with_name("train_synthetic_then_real.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = args.output_dir / "chunks.jsonl"
    history: list[dict[str, object]] = []
    chunk_start = args.start_index
    while chunk_start < args.total_steps:
        chunk_count = min(args.chunk_steps, args.total_steps - chunk_start)
        checkpoint = args.output_dir / "synthetic_pretrained.pt"
        command = [
            sys.executable,
            str(entry),
            "--device",
            args.device,
            "--synthetic-steps",
            str(chunk_count),
            "--synthetic-start-index",
            str(chunk_start),
            "--real-steps",
            "0",
            "--output-dir",
            str(args.output_dir),
            "--target-loss",
            str(args.target_loss),
            "--target-patience",
            str(args.target_patience),
        ]
        if checkpoint.is_file():
            command += ["--resume-from", str(checkpoint)]
        if args.bounded_smoke_test:
            command.append("--bounded-smoke-test")
        if args.bf16_weights:
            command += ["--bf16-weights", "--learning-rate", "1e-2"]
        attempt = 0
        completed = None
        while True:
            attempt += 1
            completed = subprocess.run(command, check=False)
            record = {
                "chunk_start": chunk_start,
                "chunk_steps": chunk_count,
                "attempt": attempt,
                "exit_code": completed.returncode,
                "checkpoint_exists": checkpoint.is_file(),
                "checkpoint_size": checkpoint.stat().st_size if checkpoint.is_file() else None,
            }
            history.append(record)
            with progress.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if completed.returncode == 0:
                break
            if attempt >= args.max_chunk_retries:
                print(json.dumps({"error": "chunk failed after retries", **record}, sort_keys=True))
                return completed.returncode
            print(
                json.dumps(
                    {"retry": True, "chunk_start": chunk_start, "attempt": attempt},
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(args.settle_seconds)
        chunk_start += chunk_count
        if chunk_start < args.total_steps:
            time.sleep(args.settle_seconds)
    print(
        json.dumps(
            {
                "status": "complete",
                "chunks": len(history),
                "total_steps": args.total_steps,
                "views_trained": args.total_steps * 2,
                "warm_up_views_required": 4096,
                "history": history,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
