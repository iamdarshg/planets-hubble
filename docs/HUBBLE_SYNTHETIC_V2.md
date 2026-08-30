# Hubble Synthetic V2

**Status:** implemented bounded research path; R5 calibration integration remains external

This document describes the realism upgrade after the R0 synthetic-generator
audit. The old `SyntheticGenerator` remains available for small deterministic
tests. `HubbleSyntheticV2` is the opt-in path for parent-conditioned examples.

## Data flow

```text
MAST manifest / FITS loader
          |
          v
RealObservationParent
  exact exposure windows, WCS, DQ, uncertainty, detector state
          |
          +--> ObservationScheduleSampler
          |      exact replay or whole-visit block bootstrap
          |
          +--> PopulationSampler
          |      coupled star -> orbit -> derived duration
          |
          +--> PsfProvider
          |      empirical kernel when supplied, optical fallback otherwise
          |
          +--> RealParentInjector
                 astrophysical perturbation only
          |
          v
iter_parented_synthetic_training_batches
          |
          v
AstroMamba-H input contract
```

The parent is the source of truth for observation mode and cadence. A
population draw cannot change a UVIS parent into an IR or Kepler observation.
The injection produces a null/injected pair that shares the parent DQ and
uncertainty arrays. Transit times are returned as `transit_times_bjd_tdb` and
the per-exposure injected depth is returned as `relative_flux_drop`.

## Implemented API

MAST discovery has explicit timeseries helpers for both named targets and sky
patches. `ManifestParentLoader` consumes normalized manifest records while
`FitsManifestParentLoader` lazily reads SCI/ERR/DQ extensions from FITS files;
both preserve product identity, exposure timing, WCS/pointing, observer
geometry, angular size, and provenance. Prepared 720x1280 NPZ parents are used
by the low-RSS local worker. A timing converter is mandatory when a manifest
is not already in BJD_TDB; the local preparation probe converts UTC MJD to TDB
JD but does not claim a full barycentric light-time correction.

```python
from synthetic import HubbleSyntheticV2, ObservationScheduleSampler

schedule = ObservationScheduleSampler(parent).sample()
result = HubbleSyntheticV2(seed=7).generate(parent, sample_index=0)

result.population.star
result.population.planet
result.injection.null
result.injection.injected
result.injection.transit_times_bjd_tdb
```

Parent arrays are copied and made read-only at construction. The injector then
copies them again for its paired outputs, so generating a sample cannot mutate
the archival parent in memory.

## Realism ladder

| Tier | Capability | Repository status |
| --- | --- | --- |
| R0 | Gaussian/analytic bounded synthetic generator | Existing compatibility path |
| R1 | Coupled population priors and exact parent cadence | Implemented |
| R2 | Empirical-first PSF API plus wavelength/focus/position optical fallback | Implemented; empirical kernels are caller-supplied external assets |
| R3 | UVIS spatial CTE approximation; IR MULTIACCUM, ramp fitting, nonlinearity, cosmic rays, saturation, persistence history | Implemented as explicitly bounded approximations |
| R4 | Astrophysical transit injection into real loaded parent arrays | Implemented |
| R5 | RAW/IMA injection, CRDS reference files, `calwf3`, AstroDrizzle, validation against matching MAST products | Not bundled; requires external STScI software, reference files, and archival data |

The labels `physics_fallback`, `real_parent_injection`, and `MULTIACCUM`
describe the path used; they do not claim that the fallback is a calibrated
HST detector model. No large FITS files, empirical PSF libraries, CRDS files,
or downloaded archive products belong in this repository.

## Detector boundary

WFC3/UVIS and WFC3/IR are separate classes because they have different causal
readout paths. UVIS models spatial charge-transfer trailing and propagates
saturation/cosmic-ray DQ bits. IR accumulates nondestructive reads and fits a
slope; persistence is driven by prior fluence and elapsed time through a
bounded `DetectorHistory` rather than by telescope breathing.

These are useful contracts and controlled ablations. They are not substitutes
for the official calibration pipeline. STScI documents that `calwf3` has
separate UVIS and IR processing tasks and uses raw data, telemetry, and
reference files; see the [WFC3 calibration pipeline documentation](https://hst-docs.stsci.edu/wfc3dhb/chapter-3-wfc3-data-calibration/3-1-the-calwf3-data-processing-pipeline)
and [WFC3 file-structure documentation](https://hst-docs.stsci.edu/wfc3dhb/chapter-2-wfc3-data-structure/2-2-wfc3-file-structure).

STScI also documents WFC3 empirical PSF libraries and their dependence on
detector position and focus. The [WFC3 photometry documentation](https://hst-docs.stsci.edu/wfc3dhb/chapter-9-wfc3-data-analysis/9-1-photometry)
is the reference for populating `EmpiricalPsfLibrary` from external assets.

## Training policy

Use R0-R3 for bounded pretraining and unit tests. Use R4 parent injections for
late synthetic pretraining and controlled sensitivity studies. Keep real HST
products outside Git and reserve them for post-training fine-tuning and
source/observation-lineage-held-out evaluation. R5 is a separate deployment
milestone, not an implication of the current tests.

The parent-aware stream is lazy, but a full model raster is inherently large:
one sample has shape `[1, visits, steps, 6, 720, 1280]`. Use one sample at a
time, avoid dataset-wide caching, and enforce the repository’s local 1.8 GiB
RSS and 5 GiB storage limits.

For this reason `examples/train_isolated_gpu.py` and
`examples/finetune_real_isolated.py` run one optimizer update per fresh worker.
That process boundary prevents CUDA allocator/workspace growth from turning a
multi-step exploratory run into a host-RSS violation. The synthetic stream
uses paired null/injected views; real-parent fine-tuning and holdout scoring
use the same parent loader and never write the known flux-drop label into the
model raster.
