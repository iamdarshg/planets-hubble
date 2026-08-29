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
37 passed
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
process RSS:            approximately 1.81 GiB
storage written:        0 bytes
```

This is a functional optimization smoke, not convergence evidence. It shows
that the research-size model can consume a real synthetic full-resolution
bundle and update its parameters on the local GPU under the available VRAM.

## Resource-cap result

The requested caps are 720 MB process RSS and 5 GB storage. Streaming and
synthetic-data stress checks are within those limits. The CUDA optimizer
smoke is functionally successful but is not RSS-cap compliant on this
Windows/PyTorch runtime:

```text
measured tiny GPU-smoke RSS: approximately 1.54 GiB
RSS cap:                720 MiB
resource violation:     rss
storage written:        0 bytes
```

The research synthetic-training smoke reached approximately 1.81 GiB process
RSS and is therefore also outside the requested host-RSS cap.

The excess is present even before a substantial model is used: importing
PyTorch is approximately 499 MB RSS on this machine, and CUDA runtime/kernel
initialization adds a large host-resident footprint. The current harness
therefore reports `rss_within_cap=false` and includes `"rss"` in
`resource_cap_violations`; it does not silently call the run cap-compliant.

The next resource-constrained deployment decision is to run CUDA validation
in an environment whose PyTorch/CUDA host footprint fits the 720 MB budget,
or to revise the host-RSS cap separately from the GPU-memory cap. The current
GPU evidence remains valid as a functional local-GPU smoke test, not as proof
of compliance with the requested host-RSS ceiling.
