# Planets-Hubble

Planets-Hubble is a research prototype for wavelength-aware, multimodal
exoplanet-candidate discovery in irregular astronomical observations. It is
Hubble-first, but its normalized input contract is intended to accept data
from other observatories when the wavelength, timing, geometry, uncertainty,
and calibration information are available.

This repository currently provides three connected layers:

1. MAST REST discovery and normalized observation manifests.
2. Bounded streaming plus lazy synthetic pretraining batches.
3. Parent-conditioned Hubble Synthetic V2: exact cadence replay, coupled
   population draws, empirical-first PSFs, UVIS/IR detector contracts, and
   real-parent transit injection.
4. AstroMamba-H model construction, heatmap/orbit outputs, and local CUDA
   validation.

The project is not an exoplanet-confirmation system. A model score is a
candidate-ranking signal. Scientific confirmation requires independent data
and domain review.

## Quick start

From the repository root:

```text
python -m pip install numpy torch pytest psutil
$env:PYTHONPATH = "$PWD\src"
pytest -q
```

The current test suite covers dataset discovery, bounded streaming, synthetic
physics/nuisance layers, model shapes, training guards, and synthetic-to-model
integration.

## Synthetic pretraining

The standalone generator is intentionally small by default for fast tests.
For a model-ready sample, use a full-resolution configuration:

```python
from synthetic import SyntheticConfig
from training import iter_synthetic_training_batches

config = SyntheticConfig(
    seed=23,
    visits=1,
    local_steps=1,
    raster_height=720,
    raster_width=1280,
)

stream = iter_synthetic_training_batches(
    config,
    sample_count=1000,
    device="cuda",
)
```

The iterator generates one bundle at a time, alternates injected and null
views, preserves uncertainty and missingness, and does not cache the full
training set. Synthetic observations are for pretraining and controlled
ablation. Real Hubble products are reserved for post-training/fine-tuning and
held-out evaluation; large real FITS files should remain in external storage,
with manifests, hashes, and provenance retained in the repository.

For the realism-upgraded path, construct a `RealObservationParent` from loaded
MAST/FITS products and use `HubbleSyntheticV2` or
`iter_parented_synthetic_training_batches`. This path replays the parent’s
actual exposure windows and detector mode, and changes only the astrophysical
injection. See [`docs/HUBBLE_SYNTHETIC_V2.md`](docs/HUBBLE_SYNTHETIC_V2.md) for
the R0-R5 boundary. The repository does not claim to bundle `calwf3`, CRDS
reference files, AstroDrizzle, or large empirical PSF/archive assets.

## Model and GPU smoke tests

The routine bounded smoke uses the tiny configuration:

```text
python examples/synthetic_model_smoke.py --device cuda
```

The research-size configuration is approximately 84M parameters and performs
one full-resolution synthetic optimizer step:

```text
python examples/synthetic_model_smoke.py --device cuda --research
```

The research configuration currently measures `84,004,564` parameters and
uses BF16 AMP on the validated local RTX 4060 Laptop GPU. The smoke proves
finite forward/backward/optimizer behavior; it does not prove convergence,
scientific sensitivity, calibrated probabilities, or an exoplanet discovery.

## Data discovery and streaming

The MAST client supports named-target and sky-patch workflows. Discovery writes
normalized manifest records containing observation/product identifiers,
download URIs, time bounds, exposure duration, wavelength/passband metadata,
WCS/spatial footprint, observer state, pointing, coverage, and quality fields.

`dataset.StreamingDataset` then reads a JSONL manifest lazily. Use a custom
loader for FITS, Zarr, or remote object stores; the built-in loader supports
inline values and `.npy`/`.npz` arrays.

## Resource boundary

The intended local-development limits are:

```text
host RSS: 1.8 GiB
storage:  5 GiB
```

Synthetic generation and storage checks stay within those limits. The
research-mode smoke constructs the model directly on CUDA and has measured
approximately 1.67 GiB process RSS on the validated Windows/PyTorch/CUDA
installation, within the revised host-RSS ceiling. See
[`docs/LOCAL_GPU.md`](docs/LOCAL_GPU.md) for the measured boundary.

## Project documents

- [`SPEC.md`](SPEC.md): top-level system and model specification.
- [`docs/SYNTHETIC_DATA.md`](docs/SYNTHETIC_DATA.md): simulator curriculum,
  nuisance taxonomy, timing policy, and real-data holdout policy.
- [`docs/HUBBLE_SYNTHETIC_V2.md`](docs/HUBBLE_SYNTHETIC_V2.md): implemented
  parent-conditioned simulator path and R0-R5 capability boundary.
- [`docs/LOCAL_GPU.md`](docs/LOCAL_GPU.md): executable local-GPU evidence and
  resource measurements.

## Development status

The current branch is a bounded, testable research prototype. The next
scientific steps are calibration against real Hubble products, concrete FITS
and WCS preprocessing, empirical nuisance-template replay, multi-sample
training, and held-out real-data evaluation. Synthetic results alone must not
be presented as astronomical confirmation.
