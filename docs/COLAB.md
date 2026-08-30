# Google Colab packaging (free T4)

The repository ships one monolithic notebook that runs the full bounded
lifecycle on a free Colab GPU runtime and returns results to a daemon on this
machine.  No Google Drive is used.

Open it directly:

    https://colab.research.google.com/github/iamdarshg/planets-hubble/blob/main/colab/planets_hubble_colab.ipynb

Free Colab typically provides a T4 (16 GiB VRAM).  If the runtime has no GPU,
use Runtime -> Change runtime type -> T4 GPU and re-run the notebook.

## What the notebook does

1. Clones the public repo into /content/planets-hubble and installs psutil.
2. Discovers the upload daemon from colab/upload_endpoint.txt (fetched from
   GitHub raw) and verifies /health.
3. Downloads the prepared real-parent bundle (about 21 MB) from the daemon.
4. Runs chunked synthetic pretraining: 2048 paired steps = 4096 views, the
   warm-up gate before any real data, with fresh subprocesses per chunk,
   resume from chunks.jsonl, an SSD-backed procedural cache, and a wall-clock
   guard inside the free session window.
5. Runs a synthetic learnability check on unseen seeds (decoder-off model
   matching the training checkpoint).
6. Fine-tunes on prepared real-parent manifests with paired positive/null
   optimizer steps.
7. Evaluates the held-out HD 209458 parent sequence (transfer smoke).
8. Uploads checkpoints, logs, and the run report to the local daemon.

## Resource boundaries

The notebook sets PLANETS_HUBBLE_RSS_CAP_BYTES=6 GiB and
PLANETS_HUBBLE_STORAGE_CAP_BYTES=12 GiB for the Colab VM (about 12 GiB RAM and
78 GiB ephemeral disk).  The local Windows defaults stay at the user's 1.6 GiB
host-RSS and 5 GiB storage caps; the env-override is implemented in
src/training/harness.py and covered by tests.

## Local receiver daemon

Run tools/start_colab_receiver.ps1 on this machine.  It:

1. Builds data/real prepared directories into
   artifacts/colab-uploads/bundles/real_prepared_bundle.zip.
2. Starts tools/colab_receiver.py on 127.0.0.1:8787 (hidden window).
3. Downloads/launches a free cloudflared quick tunnel (no account needed).
4. Writes colab/upload_endpoint.txt (JSON: url + token), which the notebook
   fetches from GitHub raw.
5. Verifies /health through the tunnel.

Receiver endpoints (localhost:8787):

    GET  /health
    POST /upload?token=TOKEN&name=FILE&subdir=SUBDIR   (raw body)
    GET  /uploads?token=TOKEN
    GET  /files?token=TOKEN&name=SUBDIR/FILE
    GET  /bundle?token=TOKEN&name=real_prepared_bundle.zip

Uploads land under artifacts/colab-uploads/uploads/<run-id>/.  The receiver
has no delete endpoint, no arbitrary host file read, no shell execution, and
validates every name; uploads are size-capped and storage-capped.

## Security note

colab/upload_endpoint.txt is committed to the public repo so the notebook can
auto-discover the tunnel.  Anyone with that file can push files into the
upload directory.  Treat the upload directory as disposable, stop the tunnel
when it is not needed, and never point the tunnel at anything else.  The
receiver is intentionally not a general-purpose file server.

## Resume after a lost session

Re-open the notebook, set RESUME_RUN_ID to the previous run id, and re-run
from the setup cell.  If the local checkpoint was lost, the notebook pulls it
back from the daemon through GET /files.

## Honesty

Synthetic pair scores are learnability evidence on unseen seeds.  The
held-out real-parent evaluation is a transfer smoke, not a discovery claim.
The notebook does not claim convergence, grokking, calibrated probabilities,
orbital accuracy, or an exoplanet confirmation.
