# Local GPU validation

This document records the current executable validation boundary. A passing
smoke test demonstrates that the code path runs; it does not demonstrate
scientific sensitivity, calibrated probabilities, or orbital-parameter
accuracy.

## Environment

The validation machine currently exposes:

```text
Python:         3.12.2
PyTorch:        2.9.1+cu130
CUDA runtime:   13.0
GPU:            NVIDIA GeForce RTX 4060 Laptop GPU
GPU memory:     8188 MiB
```

## Verified paths

The latest repository-wide test command passed after making the full-raster
architecture test use its no-decoder configuration (the test checks spatial
gradient reachability, not decoder output):

```text
python -m pytest -q
69 passed in 14.77s
```

`python -m compileall -q src examples tests` also completed successfully.

The real AstroMamba-H CUDA smoke path uses a bounded one-batch, one-visit,
one-local-step input with the full 720x1280 raster contract. It performs a
forward pass, finite loss check, backward pass, and optimizer step:

```text
model:                 AstroMambaH
input:                 [1, 1, 1, 6, 720, 1280]
AMP:                   bfloat16
loss_is_finite:        true
peak CUDA allocated:   approximately 72.6 MiB
storage written:       0 bytes
```

The smoke uses the deliberately tiny AstroMamba-H configuration so that it is
appropriate for a local functional check. The current research preset is
82,541,531 parameters and is construction-tested, but it has not yet been
trained to convergence.

The research preset has also completed a local-GPU inference probe using one
full-resolution frame. The current research parameter count is exactly
82,541,531:

```text
parameters:             82,541,531
input:                  [1, 1, 1, 6, 720, 1280]
AMP:                    bfloat16
heatmap:                [1, 1, 1, 16, 6, 90, 160]
finite output:          true
peak CUDA allocated:    approximately 489.78 MiB
peak CUDA reserved:     approximately 610.00 MiB
```

This is inference evidence only. It does not establish that the research model
fits a multi-batch training run or that it has learned exoplanet detection.

The research preset has now also completed a one-batch synthetic training
smoke. This uses the same full raster contract as the model, performs forward,
finite-loss validation, backward, and an optimizer step. The values below
were recorded for the current 82,541,531-parameter preset on 2026-08-30 with
single-threaded BLAS and the CUDA allocator configuration used for training:

```text
command:                python examples/synthetic_model_smoke.py --device cuda --research
configuration:          research
parameters:             82,541,531
input:                  [1, 1, 1, 6, 720, 1280]
AMP:                    bfloat16
loss_is_finite:         true
peak CUDA allocated:    3,661,529,088 bytes
process RSS:            980,860,928 bytes
storage written:        0 bytes
```

This is a functional optimization smoke, not convergence evidence. It shows
that the research-size model can consume a real synthetic full-resolution
bundle and update its parameters on the local GPU under the available VRAM and
the current 1.6 GiB host-RSS cap.

Synthetic pretraining is procedural: the training loop generates the next
counterfactual bundle while the model is running. A bounded SSD cache can
retain up to 64 generated entries for reuse, but it does not retain a
dataset-sized tensor cache in RAM. The two-phase lifecycle admits real-parent
fine-tuning only after 4,096 synthetic views have actually been consumed. That
threshold is a curriculum/order guarantee, not scientific sufficiency.

The bounded isolated-worker probes completed multiple synthetic curricula. The
latest varied-seed direct-source-photometry runs remained finite and
cap-compliant, but the fixed eight-seed diagnostic reached only 13/16 correct
at the 0.5 threshold. Pairwise injected-vs-null ranking was 8/8, so the model
has learned useful counterfactual ordering, but its absolute synthetic score
calibration is not yet reliable. This is not grokking or convergence evidence.

The v7 real-parent pass covered downloaded exposure records from seven
non-holdout HST target manifests (HD 189733, WASP-12, WASP-39, WASP-43,
HAT-P-11, GJ 1214, and prepared HD 209458), using synchronized positive/null
workers for 4 steps per target. A normalization bug was found and fixed during
this pass: MAST
``observation_start``/``observation_end`` can describe a broad visit window,
while ``exposure_duration_seconds`` describes the actual integration. The
parent contract now preserves both values and uses the explicit physical
duration for detector and model inputs. The v7 sweep remained finite; the
largest newly recorded worker RSS was 1,892,282,368 bytes and peak CUDA
allocation was 1,563,218,944 bytes. The v7 checkpoint was evaluated on the
separate two-sample held-out HD 209458 manifest. The first evaluation exposed
a source-conditioning bug: the pooled patch head overrode negative
source-specific evidence. The final event logit now keeps the persistent source
anchor as the primary evidence and bounds the pooled patch contribution. The
same held-out pair then returned:

```text
accuracy:               1.0 (2/2)
event probabilities:    0.8244618773, 0.0179862101
mean probability:       0.4212240437
```

This is a successful bounded transfer smoke on a two-sample holdout, not a
claim of Hubble exoplanet sensitivity, a calibrated probability, grokking, or
accurate orbital parameters. The holdout is too small for a scientific
performance estimate, and the current run used the sequence-summary fallback.

## Resource-cap result

The requested caps are 1.6 GiB process RSS and 5 GiB storage. Every long
training/evaluation process must report these measurements; the values below
are a gate, not a claim that an unmeasured future run is compliant:

```text
measured research RSS: 980,860,928 bytes for the full research smoke (1.6 GiB cap)
measured isolated-step RSS: 1,892,282,368 bytes for the largest v7 worker
RSS cap:                1.60 GiB
resource violation:     none in the recorded smoke/isolated-step runs
stored artifacts plus real data: 899,189,524 bytes
storage cap:             5 GiB
```

The prior CPU-to-CUDA construction path reached approximately 1.88 GiB and
was outside the then-current cap. Research mode now constructs the model
directly on CUDA, avoiding that migration overhead. Those reports were
historical; the current research smoke measurement above was taken under the
1.6 GiB cap.

The current full research smoke (82,541,531 parameters) measured finite loss,
3,661,529,088 bytes peak CUDA allocation, and 980,860,928 bytes process RSS.
The procedural synthetic pretraining job was also observed mid-run at
approximately 0.61 GiB process RSS with 3.85 GiB CUDA in use and an actively
growing SSD cache; it stays under the 1.6 GiB cap.
The v7 paired real-parent workers were measured under the previous 1.8 GiB
cap, with a maximum of 1,892,282,368 bytes (historical). Current storage
accounting is 806,928,292 bytes under
`artifacts/` and 92,261,232 bytes under `data/`, for 899,189,524 bytes total.
Checkpoint snapshots are overwritten or removed after each bounded step; the
old redundant checkpoint pile is not retained.

The harness continues to report the measured RSS and any violations rather
than assuming compliance. The cap is runtime- and process-specific; future
changes to PyTorch, CUDA, model construction, or batch size require another
measurement.

## Full-parent sequence fallback and counterfactual integrity

Flattening all four full-resolution parent exposures through the research
model at once exceeded the then-current host cap on this Windows/CUDA build
(historical; measured under the previous 1.8 GiB cap):

```text
full parent sequence, dense decoder:   1,971,089,408 bytes RSS (over cap)
full parent sequence, summary fallback: 1,878,683,648 bytes RSS (within cap)
summary fallback peak CUDA:            1,462,091,776 bytes
```

The `--sequence-summary` path therefore consumes every parent exposure, but
reduces the raster cadence to one spatial summary before the model. It is a
resource-bounded fallback, not equivalent to the intended local-time plus
long-baseline Mamba sequence. The full temporal path remains a required future
optimization before making orbital claims.

Real-parent labels now force a `planet_transit` for even sample indices and a
null counterfactual for odd indices. The injector no longer clips calibrated
negative FLT/FLC background pixels to zero; its injected-minus-null delta is
non-positive and localized to the source PSF. Prepared parents map the target
RA/Dec through the SCI WCS into the crop and record `source_x`, `source_y`, and
the mapping method in the manifest. Missing WCS falls back to the crop center
with an explicit provenance warning.

Because two full-resolution counterfactual batches do not fit simultaneously
under the host cap, `finetune_real_isolated.py --paired` launches synchronized
positive and null workers from the same checkpoint and averages their model
updates on CPU. This approximates one paired optimizer step without a label
ordering bias while keeping each GPU worker below the cap. The source-aware
event combination is covered by an architecture regression test and was
rechecked against the v7 holdout pair after the real sweep.

The current evidence is therefore an executable, cap-compliant data/model
path, a corrected counterfactual generator, useful but not-yet-grokking
synthetic ranking, and a positive two-sample unseen-real counterfactual smoke.
It is still not evidence of a confirmed exoplanet, convergence, grokking,
calibrated probabilities, or accurate orbital parameters. The held-out test is
an injected/null pair on an unseen real parent, not an independent discovery
claim.
