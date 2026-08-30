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

The latest repository-wide test command passed:

```text
python -m pytest -q
66 passed in 16.41s
```

```text
pytest -q
66 passed in 16.41s
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
latest coordinate-aware source+event curriculum completed 32 synthetic views;
loss ranged approximately 0.57--1.49, with most later injected views below
0.75. All updates were finite and cap-compliant, but this is still not
grokking/convergence evidence because the loss is not stably below the target.
The leakage-free real repeat was followed by evaluation on four unseen-parent
exposure indices: each returned 1/2 correct (0.50 accuracy), with event
probabilities near 0.36. This is a controlled integration result, not a claim
of Hubble exoplanet sensitivity; the holdout contains only four parent
exposures and synthetic counterfactual labels.

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
latest coordinate-aware BF16 synthetic and real-parent runs.

The harness continues to report the measured RSS and any violations rather
than assuming compliance. The cap is runtime- and process-specific; future
changes to PyTorch, CUDA, model construction, or batch size require another
measurement.
