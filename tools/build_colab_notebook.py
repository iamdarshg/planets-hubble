"""Generate the monolithic Colab notebook for planets-hubble."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "planets_hubble_colab.ipynb"

CELLS: list[dict[str, object]] = []


def md(source: str) -> None:
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    })


def code(source: str) -> None:
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    })


md("""# Planets-Hubble - Colab T4 monolith

One notebook for the full bounded lifecycle:

1. procedural synthetic pretraining (SSD-backed cache, chunked, resumable),
2. real-parent HST fine-tuning on prepared WFC3/IR parents,
3. held-out real-parent evaluation,
4. upload of checkpoints, logs, and the run report back to your local daemon.

No Google Drive is used.  Results travel through a free cloudflared tunnel to
a receiver process on the host machine and land under
artifacts/colab-uploads/uploads/<run-id>.

## Honesty contract

- Synthetic results are learnability evidence only.
- Real-parent results are transfer smokes, never discovery.
- This notebook cannot claim convergence, grokking, calibrated probabilities,
  orbital accuracy, or an exoplanet confirmation by itself.

## GPU note

Free Colab usually provides a T4 (16 GiB VRAM).  If the runtime has no GPU,
use Runtime -> Change runtime type -> T4 GPU and re-run this notebook.
""")

md("""## What happens

- Cell 4 clones the public repo into /content/planets-hubble.
- Cells 6-7 discover the upload daemon and download the prepared real dataset.
- Cells 8-11 run chunked synthetic pretraining: 2048 paired steps = 4096 views
  (the warm-up gate before any real data), with fresh subprocesses per chunk so
  CUDA workspace growth cannot stall the run, and a wall-clock guard that stays
  inside the free Colab session window.
- Cells 12-14 run a synthetic learnability check on unseen seeds, then
  fine-tune on real parents with paired positive/null updates, then evaluate
  the held-out HD 209458 parent sequence.
- Cell 15 uploads the report and checkpoints to the daemon.

If the session dies, re-run the notebook: it resumes from chunks.jsonl and the
checkpoint (and can pull a checkpoint back from the daemon if you set
RESUME_RUN_ID).
""")

code("""import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path("/content/planets-hubble")
ROOT.mkdir(parents=True, exist_ok=True)

# Colab VM limits are roughly 12 GiB RAM and ~78 GiB ephemeral disk.  The
# local Windows host keeps its own 1.6 GiB RSS / 5 GiB storage defaults.
os.environ["PLANETS_HUBBLE_RSS_CAP_BYTES"] = str(6 * 1024 ** 3)
os.environ["PLANETS_HUBBLE_STORAGE_CAP_BYTES"] = str(12 * 1024 ** 3)

print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("gpu", torch.cuda.get_device_name(0), round(props.total_memory / (1024 ** 2)), "MiB")
except Exception as exc:  # pragma: no cover
    print("torch import failed:", exc)

total, used, free = shutil.disk_usage("/content")
print("disk_free_gb", round(free / (1024 ** 3), 1))
""")

code("""import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "No CUDA GPU in this runtime.  Use Runtime -> Change runtime type -> "
        "T4 GPU, then re-run the notebook."
    )
print("GPU OK")
""")

code("""if not (ROOT / ".git").is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/iamdarshg/planets-hubble.git", str(ROOT)],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(ROOT), "pull", "--ff-only"], check=False)
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT / "src")
sys.path.insert(0, str(ROOT / "src"))
print("repo", ROOT)
""")

code("""subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "psutil"],
    check=False,
)
import psutil
print("psutil", psutil.__version__)
""")

code("""ENDPOINT_URL = "https://raw.githubusercontent.com/iamdarshg/planets-hubble/main/colab/upload_endpoint.txt"
UPLOAD_URL = ""    # optional manual override, e.g. "https://abc.trycloudflare.com"
UPLOAD_TOKEN = ""  # optional manual override

def fetch_endpoint():
    if UPLOAD_URL and UPLOAD_TOKEN:
        return {"url": UPLOAD_URL.rstrip("/"), "token": UPLOAD_TOKEN}
    try:
        with urllib.request.urlopen(ENDPOINT_URL, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {"url": str(payload["url"]).rstrip("/"), "token": str(payload["token"])}
    except Exception as exc:
        print("endpoint auto-discovery failed:", exc)
        return None

ENDPOINT = fetch_endpoint()
if ENDPOINT:
    try:
        with urllib.request.urlopen(ENDPOINT["url"] + "/health", timeout=20) as response:
            health = json.loads(response.read().decode("utf-8"))
        print("daemon health:", health)
    except Exception as exc:
        print("daemon unreachable:", exc)
        ENDPOINT = None
print("ENDPOINT_OK", ENDPOINT is not None)
""")

code("""BUNDLE_NAME = "real_prepared_bundle.zip"

def download_bundle():
    if ENDPOINT is None:
        print("no daemon endpoint; skipping real-data download")
        return False
    url = (ENDPOINT["url"] + "/bundle?token=" + urllib.parse.quote(ENDPOINT["token"])
           + "&name=" + urllib.parse.quote(BUNDLE_NAME))
    destination = ROOT / "data" / "real" / BUNDLE_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".zip.part")
    try:
        with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        partial.replace(destination)
        with zipfile.ZipFile(destination) as archive:
            archive.extractall(ROOT / "data" / "real")
        prepared = sorted((ROOT / "data" / "real").glob("*_prepared"))
        print("prepared bundles:", [p.name for p in prepared])
        return True
    except Exception as exc:
        print("bundle download failed:", exc)
        partial.unlink(missing_ok=True)
        return False

REAL_DATA_READY = download_bundle()
""")
code("""RUN_ID = "colab-" + time.strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = ROOT / "artifacts" / "training" / RUN_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUTPUT_DIR / "synthetic_pretrained.pt"
REAL_CHECKPOINT = OUTPUT_DIR / "real_finetuned.pt"

SYNTHETIC_PAIRED_STEPS = 2048   # each paired step = 1 null + 1 injected view -> 4096 views
CHUNK_STEPS = 40
TARGET_LOSS = 0.05
TARGET_PATIENCE = 3
MAX_CHUNK_RETRIES = 3
SETTLE_SECONDS = 20
BF16_WEIGHTS = True
LEARNING_RATE = 1e-2
CUDNN_OFF = False                # Linux host keeps cuDNN on; RSS cap is 6 GiB here
MAX_WALL_SECONDS = 8 * 3600      # stay inside the free Colab session window
UPLOAD_CHECKPOINT_EVERY_STEPS = 320  # upload the checkpoint periodically
RESUME_RUN_ID = ""               # optional: pull a prior checkpoint from the daemon
print("RUN_ID", RUN_ID)
""")

code("""def upload_file(remote_name, local_path, subdir=None):
    local_path = Path(local_path)
    if ENDPOINT is None or not local_path.is_file():
        return None
    subdir = subdir or RUN_ID
    with local_path.open("rb") as handle:
        data = handle.read()
    url = (ENDPOINT["url"] + "/upload?token=" + urllib.parse.quote(ENDPOINT["token"])
           + "&name=" + urllib.parse.quote(remote_name)
           + "&subdir=" + urllib.parse.quote(subdir))
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/octet-stream")
    request.add_header("Content-Length", str(len(data)))
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print("upload failed", remote_name, exc)
        return None

def upload_json(remote_name, payload, subdir=None):
    if ENDPOINT is None:
        return None
    subdir = subdir or RUN_ID
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    url = (ENDPOINT["url"] + "/upload?token=" + urllib.parse.quote(ENDPOINT["token"])
           + "&name=" + urllib.parse.quote(remote_name)
           + "&subdir=" + urllib.parse.quote(subdir))
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Content-Length", str(len(data)))
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print("upload failed", remote_name, exc)
        return None

def download_file(remote_name, local_path, timeout=600):
    if ENDPOINT is None:
        return False
    local_path = Path(local_path)
    url = (ENDPOINT["url"] + "/files?token=" + urllib.parse.quote(ENDPOINT["token"])
           + "&name=" + urllib.parse.quote(remote_name))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, local_path.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        return True
    except Exception as exc:
        print("download failed", remote_name, exc)
        return False

print("helpers ready")
""")

code("""def completed_steps():
    progress = OUTPUT_DIR / "chunks.jsonl"
    if not progress.is_file():
        return 0
    total = 0
    for line in progress.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except Exception:
            continue
        if record.get("exit_code") == 0:
            total += int(record.get("chunk_steps", 0))
    return total

START_INDEX = completed_steps()
if START_INDEX > 0 and not CHECKPOINT.is_file() and RESUME_RUN_ID:
    ok = download_file(RESUME_RUN_ID + "/synthetic_pretrained.pt", CHECKPOINT)
    print("remote checkpoint restore", ok)
if START_INDEX > 0 and not CHECKPOINT.is_file():
    print("WARNING: chunks.jsonl exists but checkpoint is missing; start from zero")
    START_INDEX = 0
print("START_INDEX", START_INDEX)
""")

code("""def run_chunk(start, count):
    command = [
        sys.executable, "examples/train_synthetic_then_real.py",
        "--device", "cuda",
        "--synthetic-steps", str(count),
        "--synthetic-start-index", str(start),
        "--real-steps", "0",
        "--output-dir", str(OUTPUT_DIR),
        "--target-loss", str(TARGET_LOSS),
        "--target-patience", str(TARGET_PATIENCE),
    ]
    if CHECKPOINT.is_file():
        command += ["--resume-from", str(CHECKPOINT)]
    if BF16_WEIGHTS:
        command += ["--bf16-weights", "--learning-rate", str(LEARNING_RATE)]
    if CUDNN_OFF:
        command.append("--cudnn-off")
    completed = subprocess.run(command, capture_output=True, text=True)
    summary = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("{"):
            try:
                summary = json.loads(line)
            except Exception:
                pass
            break
    record = {
        "chunk_start": start,
        "chunk_steps": count,
        "exit_code": completed.returncode,
        "summary": summary,
    }
    with (OUTPUT_DIR / "chunks.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + chr(10))
    if completed.returncode != 0:
        print(completed.stdout[-2000:])
        print(completed.stderr[-2000:])
    return record

started = time.time()
history = []
index = START_INDEX
while index < SYNTHETIC_PAIRED_STEPS:
    if time.time() - started > MAX_WALL_SECONDS:
        print("wall-clock guard reached; stopping after", index, "steps")
        break
    count = min(CHUNK_STEPS, SYNTHETIC_PAIRED_STEPS - index)
    record = None
    for attempt in range(1, MAX_CHUNK_RETRIES + 1):
        record = run_chunk(index, count)
        if record["exit_code"] == 0:
            break
        print("chunk failed (attempt", attempt, "of", MAX_CHUNK_RETRIES, ")")
        time.sleep(SETTLE_SECONDS)
    history.append(record)
    if record is None or record["exit_code"] != 0:
        print("chunk failed after retries at index", index)
        break
    index += count
    upload_json("progress.jsonl", {
        "run_id": RUN_ID,
        "index": index,
        "total": SYNTHETIC_PAIRED_STEPS,
        "last": record["summary"],
    })
    if index % UPLOAD_CHECKPOINT_EVERY_STEPS == 0 and CHECKPOINT.is_file():
        upload_file("synthetic_pretrained.pt", CHECKPOINT)
    elapsed_min = round((time.time() - started) / 60.0, 1)
    print("progress", index, "/", SYNTHETIC_PAIRED_STEPS, "elapsed_min", elapsed_min)

views_trained = index * 2
synthetic_finished = index >= SYNTHETIC_PAIRED_STEPS
print("views_trained", views_trained, "gate_4096_open", views_trained >= 4096, "finished", synthetic_finished)
""")
code("""from dataclasses import replace

import torch

from model import research_config
from synthetic import SyntheticConfig
from training import AstroMambaHTrainingAdapter, iter_paired_synthetic_training_batches, resolve_device
from training.pipeline import _split_batch

device = resolve_device("cuda")
config = replace(research_config(), decode_heatmaps=False)
with torch.device(device):
    model = AstroMambaHTrainingAdapter(config=config)
model = model.to(device, dtype=torch.bfloat16)
state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(state["model"] if "model" in state else state)
model.eval()

def sigmoid(logit):
    return 1.0 / (1.0 + math.exp(-float(logit)))

synthetic_pair_scores = []
for seed in (101, 202, 303):
    pair = next(iter_paired_synthetic_training_batches(
        SyntheticConfig(
            seed=seed,
            visits=1,
            local_steps=1,
            raster_height=720,
            raster_width=1280,
            wavelength_nm=(450.0, 650.0, 1000.0),
        ),
        sample_count=1,
        device="cpu",
        start_index=0,
    ))
    with torch.inference_mode():
        for view, batch in enumerate(_split_batch(pair)):
            batch = batch.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch)
            logit = float(output["global_event_logits"].reshape(-1)[0].float().cpu())
            target = float(batch.target.reshape(-1)[0].cpu())
            synthetic_pair_scores.append({
                "seed": seed,
                "view": view,
                "target": target,
                "logit": logit,
                "probability": sigmoid(logit),
            })
    del pair
    torch.cuda.empty_cache()
for row in synthetic_pair_scores:
    print(row)
del model
torch.cuda.empty_cache()
""")

code("""REAL_STEPS = 4
train_manifests = []
holdout_manifest = ROOT / "data" / "real" / "hd209458_holdout_prepared" / "manifest.json"
prepared = sorted((ROOT / "data" / "real").glob("*_prepared"))
for directory in prepared:
    if "holdout" in directory.name:
        continue
    manifest = directory / "manifest.json"
    if manifest.is_file():
        train_manifests.append(manifest)

real_finetuned = False
if views_trained >= 4096 and REAL_DATA_READY and CHECKPOINT.is_file() and train_manifests:
    shutil.copy2(CHECKPOINT, REAL_CHECKPOINT)
    for target_index, manifest in enumerate(train_manifests):
        print("finetuning", manifest.parent.name)
        command = [
            sys.executable, "examples/finetune_real_isolated.py",
            "--manifest", str(manifest),
            "--checkpoint", str(REAL_CHECKPOINT),
            "--steps", str(REAL_STEPS),
            "--paired",
            "--skip-dense-heatmaps",
            "--learning-rate", "1e-2",
        ]
        if target_index == 0:
            command += ["--input-checkpoint", str(CHECKPOINT)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            print(completed.stdout[-3000:])
            print(completed.stderr[-3000:])
            print("real fine-tune failed for", manifest.parent.name)
            break
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith("{"):
                try:
                    upload_json("real_progress.jsonl", json.loads(line))
                except Exception:
                    pass
                break
    else:
        real_finetuned = REAL_CHECKPOINT.is_file()
        print("real fine-tune complete:", real_finetuned)
else:
    print("real phase skipped:",
          "gate", views_trained >= 4096,
          "data", REAL_DATA_READY,
          "checkpoint", CHECKPOINT.is_file(),
          "manifests", len(train_manifests))
""")

code("""eval_result = None
eval_source = REAL_CHECKPOINT if real_finetuned else CHECKPOINT
if REAL_DATA_READY and holdout_manifest.is_file() and eval_source.is_file():
    print("held-out eval", holdout_manifest, "checkpoint", eval_source)
    completed = subprocess.run(
        [
            sys.executable, "examples/evaluate_real_parent.py",
            "--manifest", str(holdout_manifest),
            "--checkpoint", str(eval_source),
            "--sequence-summary",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith("{"):
                try:
                    eval_result = json.loads(line)
                except Exception:
                    pass
                break
    else:
        print(completed.stdout[-3000:])
        print(completed.stderr[-3000:])
print("held_out_eval", eval_result)
""")

code("""report = {
    "run_id": RUN_ID,
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "synthetic_paired_steps": index,
    "synthetic_views_trained": views_trained,
    "synthetic_gate_4096_open": views_trained >= 4096,
    "synthetic_finished": synthetic_finished,
    "synthetic_checkpoint_bytes": CHECKPOINT.stat().st_size if CHECKPOINT.is_file() else None,
    "real_finetuned": real_finetuned,
    "real_checkpoint_bytes": REAL_CHECKPOINT.stat().st_size if REAL_CHECKPOINT.is_file() else None,
    "synthetic_pair_scores": synthetic_pair_scores,
    "held_out_real_eval": eval_result,
    "endpoint": ENDPOINT["url"] if ENDPOINT else None,
    "honesty": "synthetic learnability and real-parent transfer smoke only; not a discovery or calibration claim",
}
with (OUTPUT_DIR / "run_report.json").open("w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

upload_json("run_report.json", report)
upload_file("synthetic_pretrained.pt", CHECKPOINT)
upload_file("chunks.jsonl", OUTPUT_DIR / "chunks.jsonl")
if REAL_CHECKPOINT.is_file():
    upload_file("real_finetuned.pt", REAL_CHECKPOINT)
if eval_result is not None:
    upload_json("held_out_eval.json", eval_result)
print("run report uploaded to", ENDPOINT["url"] if ENDPOINT else "no endpoint")
print("local landing: artifacts/colab-uploads/uploads/" + RUN_ID)
""")

md("""## Wrap-up

- The run report and checkpoints are now on the host machine under
  artifacts/colab-uploads/uploads/<run-id> (subdir equals the run id printed
  above).
- Synthetic pair scores are learnability evidence on unseen seeds.
- The held-out real eval is a transfer smoke, not a discovery claim.
- To resume after a session loss: re-open this notebook, set RESUME_RUN_ID to
  the previous run id, and re-run from the setup cell.
""")

def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": CELLS,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    OUT.write_text(json.dumps(notebook, indent=1) + chr(10), encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(build())
