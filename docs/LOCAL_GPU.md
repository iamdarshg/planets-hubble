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

The repository-wide test command currently passes:

```text
pytest -q
38 passed
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
appropriate for a local functional check. The measured research preset is
approximately 84.0M parameters and is construction-tested, but it has not
yet been trained to convergence.

The research preset has also completed a local-GPU inference probe using one
full-resolution frame:

```text
parameters:             84,004,564
input:                  [1, 1, 1, 6, 720, 1280]
AMP:                    bfloat16
heatmap:                [1, 1, 1, 16, 6, 90, 160]
finite output:          true
peak CUDA allocated:    approximately 489.78 MiB
peak CUDA reserved:     approximately 610.00 MiB
```

This is inference evidence only. It does not establish that the 84M model
fits a multi-batch training run or that it has learned exoplanet detection.

The research preset has now also completed a one-batch synthetic training
smoke. This uses the same full raster contract as the model, performs forward,
finite-loss validation, backward, and an AdamW optimizer step:

```text
command:                python examples/synthetic_model_smoke.py --device cuda --research
configuration:          research
parameters:             84,004,564
input:                  [1, 1, 1, 6, 720, 1280]
AMP:                    bfloat16
loss_is_finite:         true
peak CUDA allocated:    approximately 1.53 GiB
process RSS:            approximately 1.67 GiB
storage written:        0 bytes
```

This is a functional optimization smoke, not convergence evidence. It shows
that the research-size model can consume a real synthetic full-resolution
bundle and update its parameters on the local GPU under the available VRAM and
the revised 1.8 GiB host-RSS cap.

## Resource-cap result

The revised requested caps are 1.8 GiB process RSS and 5 GiB storage.
Streaming, synthetic-data generation, and the direct-CUDA research smoke are
within those limits on the current validation run:

```text
measured research RSS: approximately 1.67 GiB
RSS cap:                1.80 GiB
resource violation:     none
storage written:        0 bytes
```

The prior CPU-to-CUDA construction path reached approximately 1.88 GiB and
was outside the revised cap. Research mode now constructs the model directly
on CUDA, avoiding that migration overhead. The current smoke reports
`rss_within_cap=true`.

The harness continues to report the measured RSS and any violations rather
than assuming compliance. The cap is runtime- and process-specific; future
changes to PyTorch, CUDA, model construction, or batch size require another
measurement.
