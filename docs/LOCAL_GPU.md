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
67 passed in 11.07s
```

```text
pytest -q
67 passed in 11.07s
```

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
approximately 50.2M parameters and is construction-tested, but it has not
yet been trained to convergence.

The research preset has also completed a local-GPU inference probe using one
full-resolution frame:

```text
parameters:             approximately 50,187,735
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
finite-loss validation, backward, and an AdamW optimizer step:

```text
command:                python examples/synthetic_model_smoke.py --device cuda --research
configuration:          research
parameters:             approximately 50,187,735
input:                  [1, 1, 1, 6, 720, 1280]
AMP:                    bfloat16
loss_is_finite:         true
peak CUDA allocated:    1,701,552,128 bytes
process RSS:            1,842,831,360 bytes
storage written:        0 bytes
```

This is a functional optimization smoke, not convergence evidence. It shows
that the research-size model can consume a real synthetic full-resolution
bundle and update its parameters on the local GPU under the available VRAM and
the revised 1.8 GiB host-RSS cap.

The bounded isolated-worker probes completed multiple synthetic curricula and
then 8 real-parent views (4 exposures, null/injected counterfactuals). The
latest direct-source-photometry curriculum completed 26 synthetic views before
the bounded run was stopped; on the fixed seed-23 diagnostic it produced
null/injected mean event probabilities of 0.075858/0.999981. This is strong
separation on the synthetic counterfactual, but it is not grokking or
convergence evidence because the run was intentionally short and the synthetic
distribution is not real Hubble data.

The expanded real-parent pass covered downloaded exposure records from seven
non-holdout HST target manifests (HD 189733, WASP-12, WASP-39, WASP-43,
HAT-P-11, GJ 1214, and prepared HD 209458), using synchronized positive/null
workers. A normalization bug was found and fixed during this pass: MAST
``observation_start``/``observation_end`` can describe a broad visit window,
while ``exposure_duration_seconds`` describes the actual integration. The
parent contract now preserves both values and uses the explicit physical
duration for detector and model inputs. The v6 sweep remained finite; its
largest newly recorded worker RSS was 1,891,028,992 bytes and peak CUDA
allocation was 1,563,218,944 bytes. The v6 checkpoint was evaluated on the
separate two-sample held-out HD 209458 manifest and returned:

```text
accuracy:               1.0 (2/2)
event probabilities:    0.8840392828, 0.2068940550
mean probability:       0.5454666689
```

This is a successful bounded transfer smoke on a two-sample holdout, not a
claim of Hubble exoplanet sensitivity, a calibrated probability, grokking, or
accurate orbital parameters. The holdout is too small for a scientific
performance estimate, and the current run used the sequence-summary fallback.

## Resource-cap result

The requested caps are 1.8 GiB process RSS and 5 GiB storage. Every long
training/evaluation process must report these measurements; the values below
are a gate, not a claim that an unmeasured future run is compliant:

```text
measured research RSS: 1,842,831,360 bytes for the full research smoke
measured isolated-step RSS: 1,849,352,192 bytes for a fresh worker
RSS cap:                1.80 GiB
resource violation:     none in the recorded smoke/isolated-step runs
storage written:        0 bytes
```

The prior CPU-to-CUDA construction path reached approximately 1.88 GiB and
was outside the revised cap. Research mode now constructs the model directly
on CUDA, avoiding that migration overhead. The current smoke reports
`rss_within_cap=true`.

The latest full research smoke measured 50,187,735 parameters, finite loss,
1,701,552,128 bytes peak CUDA allocation, and 1,842,831,360 bytes process RSS.
A fresh isolated optimizer step measured 1,474,002,432 bytes peak CUDA
allocation and 1,849,352,192 bytes RSS. The isolated synthetic and
real-parent workers also stayed below the cap; their largest recorded RSS
values were 1,928,249,344 and 1,911,619,584 bytes respectively in the
latest coordinate-aware BF16 synthetic and real-parent runs. The v6 sweep's
fresh paired workers were lower still, at a maximum of 1,891,028,992 bytes.

The harness continues to report the measured RSS and any violations rather
than assuming compliance. The cap is runtime- and process-specific; future
changes to PyTorch, CUDA, model construction, or batch size require another
measurement.

## Full-parent sequence fallback and counterfactual integrity

Flattening all four full-resolution parent exposures through the research
model at once exceeded the host cap on this Windows/CUDA build:

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
ordering bias while keeping each GPU worker below the cap. The six-step HD
209458 diagnostic run measured worker RSS between 1,885,118,464 and
1,891,532,800 bytes and peak CUDA allocation of 1,561,726,464 bytes. It did
not produce meaningful holdout separation: the corrected FP32 probabilities
were `0.700895` for the injected view and `0.700075` for the null view, or
0.50 accuracy at the 0.5 threshold.

The current evidence is therefore an executable, cap-compliant data/model
path, a corrected counterfactual generator, strong short-run synthetic
separation, and a positive two-sample unseen-real smoke. It is still not
evidence of a confirmed exoplanet, convergence, grokking, calibrated
probabilities, or accurate orbital parameters.
