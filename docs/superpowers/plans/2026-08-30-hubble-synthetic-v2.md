# Hubble Synthetic V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current parameter-only synthetic image path with a parent-conditioned, instrument-aware simulator that can replay real Hubble exposure windows and inject astrophysical signals into real or detector-simulated observations.

**Architecture:** Keep the existing `SyntheticGenerator` as the bounded R0 compatibility path. Add a separate V2 pipeline with typed real-parent contracts, whole-visit schedule replay, coupled stellar/planet population draws, wavelength-dependent PSF providers, and detector-specific UVIS/IR forward models. The V2 path will accept real parent arrays and DQ/history when available, while using explicitly labeled physics fallbacks when they are not.

**Tech Stack:** Python 3.12, NumPy, optional Astropy/WFC3 tools at integration boundaries, pytest, JSON-friendly manifests, and subprocess execution for optional `calwf3`/AstroDrizzle stages.

**Spec:** `SPEC.md`, `docs/SYNTHETIC_DATA.md`, and the HST WFC3 handbook references listed in those documents.

**Implementation status:** Tasks 1-5 are implemented in the opt-in V2 path and
Task 6 documentation is complete. The public API uses
`HubbleSyntheticV2`, `RealParentInjector`, `PopulationSampler`,
`ObservationScheduleSampler`, `PsfProvider`, `WFC3UVISSimulator`,
`WFC3IRSimulator`, and `iter_parented_synthetic_training_batches`. The plan's
R5 items remain external by design: real FITS loading, empirical asset
retrieval, CRDS, `calwf3`, and AstroDrizzle still require separately installed
software/data.

## Global Constraints

- Synthetic data is for pretraining; real Hubble parent exposures are reserved for post-training/fine-tuning and held-out evaluation.
- A real parent must preserve its exposure timing, WCS, detector, DQ, uncertainty, pointing, and provenance rather than being silently reduced to a generic image.
- UVIS and IR are separate detector models; IR supports nondestructive MULTIACCUM reads.
- Detector effects are causal layers and are not labeled as relativistic effects.
- Unsupported empirical calibration is labeled `calibrated_approximation` or `OOD`, never silently promoted to HST truth.
- The implementation must remain bounded for the local `1.8 GiB` RSS and `5 GiB` storage limits.
- No large FITS, PSF libraries, CRDS reference files, or downloaded archival data are committed to Git.

---

### Task 1: Real-parent and schedule contracts

**Files:**
- Create: `src/synthetic/parents.py`
- Create: `src/synthetic/schedules.py`
- Modify: `src/synthetic/__init__.py`
- Test: `tests/synthetic/test_parents_and_schedule.py`

**Interfaces:**
- `RealExposureParent` stores one exposure's detector, filter, timing, raster, uncertainty, DQ, WCS/pointing, observer state, and detector-history inputs.
- `RealObservationParent` stores an immutable tuple of real exposure parents plus source coordinates and provenance.
- `ObservationScheduleSampler(parent).sample()` returns exact parent exposure windows and preserves gaps; `block_bootstrap()` selects whole parent visits without independently jittering timestamps.

- [ ] **Step 1: Write failing contract tests**

```python
def test_parent_preserves_exposure_windows_and_provenance():
    parent = make_parent()
    assert parent.exposures[0].t_start_bjd_tdb == 2460000.1
    assert parent.exposures[0].t_end_bjd_tdb == 2460000.11
    assert parent.provenance["source"] == "MAST"

def test_schedule_replay_preserves_gaps_without_timestamp_jitter():
    schedule = ObservationScheduleSampler(make_parent()).sample()
    np.testing.assert_allclose(schedule.starts, [2460000.1, 2460002.0])
    np.testing.assert_allclose(schedule.ends, [2460000.11, 2460002.03])
    assert schedule.metadata["mode"] == "real_parent_replay"
```

- [ ] **Step 2: Run the focused tests and observe the missing-contract failure**

Run: `pytest -q tests/synthetic/test_parents_and_schedule.py`

- [ ] **Step 3: Implement immutable parent dataclasses and exact replay**

Use `np.asarray(...).copy()` only at construction boundaries, keep optional arrays nullable, validate monotonic exposure windows, and return read-only schedule arrays. Include `instrument`, `detector`, `filter_name`, `exposure_id`, `visit_id`, `t_start_bjd_tdb`, `t_mid_bjd_tdb`, `t_end_bjd_tdb`, `gain`, `read_noise`, `saturation_electrons`, `pixel_scale_arcsec`, `wcs`, `pointing`, `observer_position`, `observer_velocity`, `dq`, `science`, `uncertainty`, `focus`, `jitter`, and `provenance`.

- [ ] **Step 4: Run focused and existing synthetic tests**

Run: `pytest -q tests/synthetic/test_parents_and_schedule.py tests/synthetic`

- [ ] **Step 5: Commit**

```text
git add src/synthetic tests/synthetic/test_parents_and_schedule.py
git commit -m "Add real Hubble parent and schedule contracts"
```

### Task 2: Coupled physical population sampler

**Files:**
- Create: `src/synthetic/population.py`
- Modify: `src/synthetic/__init__.py`
- Test: `tests/synthetic/test_population.py`

**Interfaces:**
- `PopulationSampler(seed).sample_star()` returns coupled `StarParameters`.
- `PopulationSampler(seed).sample_system(star, event_type=...)` returns `PlanetParameters` or `BinaryParameters` with derived orbital quantities.
- `PopulationSampler.sample_observation(parent)` returns a `PopulationDraw` containing the physical draw, observability metadata, and simulator provenance.

- [ ] **Step 1: Write failing tests for coupled draws and hard negatives**

```python
def test_orbit_is_derived_from_mass_and_period():
    draw = PopulationSampler(seed=7).sample_system(
        StarParameters(mass_solar=1.0, radius_solar=1.0, teff_kelvin=5772.0),
        event_type="transit",
    )
    assert draw.semi_major_axis_au > 0.0
    assert draw.transit_duration_hours > 0.0
    assert draw.eccentricity < 1.0

def test_population_seed_changes_physical_system_not_only_noise():
    a = PopulationSampler(seed=1).sample_observation(make_parent())
    b = PopulationSampler(seed=2).sample_observation(make_parent())
    assert (a.star.mass_solar, a.system.period_days) != (
        b.star.mass_solar, b.system.period_days
    )
```

- [ ] **Step 2: Run the focused tests and observe the missing sampler failure**

Run: `pytest -q tests/synthetic/test_population.py`

- [ ] **Step 3: Implement coupled distributions**

Draw stellar mass first, derive radius/temperature/luminosity with bounded
relations, draw period/eccentricity/argument/inclination, derive semi-major
axis and duration, and separately draw planet radius ratio, TTV amplitude,
spot activity, and binary false-positive parameters. Record priors and
`parameter_constraint_status`; never treat the proposal distribution as an
occurrence-rate estimate.

- [ ] **Step 4: Run focused tests and existing tests**

Run: `pytest -q tests/synthetic/test_population.py tests/synthetic`

- [ ] **Step 5: Commit**

```text
git add src/synthetic tests/synthetic/test_population.py
git commit -m "Add coupled stellar and event population sampling"
```

### Task 3: Wavelength-dependent PSF provider

**Files:**
- Create: `src/synthetic/psf.py`
- Modify: `src/synthetic/__init__.py`
- Test: `tests/synthetic/test_psf.py`

**Interfaces:**
- `EmpiricalPsfLibrary.add(key, kernel)` and `.lookup(detector, filter_name, x, y, focus, wavelength_nm)` provide nearest compatible empirical kernels.
- `PhysicsPsfProvider.render(...)` produces normalized diffraction, spike, aberration, jitter, focus, and charge-diffusion kernels.
- `PsfProvider(empirical, fallback).render(...)` returns a kernel plus `psf_metadata` declaring `empirical_replay` or `physics_approximation`.

- [ ] **Step 1: Write failing tests**

```python
def test_physics_psf_has_wavelength_dependent_diffraction_and_spikes():
    provider = PhysicsPsfProvider()
    blue = provider.render(wavelength_nm=300.0, pixel_scale_arcsec=0.04, size=33)
    red = provider.render(wavelength_nm=800.0, pixel_scale_arcsec=0.04, size=33)
    assert blue.metadata["tier"] == "physics_approximation"
    assert blue.kernel.shape == (33, 33)
    assert blue.kernel.sum() == pytest.approx(1.0)
    assert not np.allclose(blue.kernel, red.kernel)
    assert np.count_nonzero(blue.kernel[16, :]) > 3

def test_empirical_psf_is_preferred_over_physics_fallback():
    library = EmpiricalPsfLibrary()
    library.add(("UVIS", "F606W", 100, 100), np.ones((5, 5), dtype=np.float32))
    result = PsfProvider(library, PhysicsPsfProvider()).render(
        detector="UVIS", filter_name="F606W", x=100, y=100,
        wavelength_nm=600.0, pixel_scale_arcsec=0.04, size=5,
    )
    assert result.metadata["tier"] == "empirical_replay"
```

- [ ] **Step 2: Run focused tests and observe the missing provider failure**

Run: `pytest -q tests/synthetic/test_psf.py`

- [ ] **Step 3: Implement empirical lookup and bounded optical fallback**

Use an Airy-like radial core with wavelength-dependent scale, diffraction
spikes, focus/asymmetry perturbations, subpixel shift, detector-position
terms, and charge diffusion. Normalize the kernel and expose provenance.
Empirical kernels remain external assets; no library is downloaded in tests.

- [ ] **Step 4: Run focused and existing tests**

Run: `pytest -q tests/synthetic/test_psf.py tests/synthetic`

- [ ] **Step 5: Commit**

```text
git add src/synthetic tests/synthetic/test_psf.py
git commit -m "Add empirical and wavelength-aware PSF providers"
```

### Task 4: Detector-specific UVIS and IR forward models

**Files:**
- Create: `src/synthetic/detectors.py`
- Modify: `src/synthetic/__init__.py`
- Test: `tests/synthetic/test_detectors.py`

**Interfaces:**
- `DetectorHistory.push(fluence, saturation, t_mid)` and `.persistence(shape, t_mid)` implement bounded fluence-history memory.
- `WFC3UVISSimulator.simulate(photons, exposure, rng, history)` returns science, uncertainty, DQ, and detector metadata with spatial CTE trailing.
- `WFC3IRSimulator.simulate(rate_image, exposure, rng, history)` returns MULTIACCUM reads, fitted slope, uncertainty, DQ, and read-level metadata.

- [ ] **Step 1: Write failing detector tests**

```python
def test_uvis_cte_is_spatial_and_dq_survives():
    photons = point_source((32, 32), x=8, y=16, electrons=1000.0)
    result = WFC3UVISSimulator(...).simulate(photons, exposure_seconds=100.0, ...)
    assert result.dq.shape == photons.shape
    assert result.metadata["cte_model"] == "spatial_trailing_approximation"
    assert result.science[:, 16].sum() > result.science[:, 16].max()

def test_ir_has_nondestructive_reads_and_history_dependent_persistence():
    history = DetectorHistory(max_exposures=2)
    simulator = WFC3IRSimulator(read_times_seconds=(0.0, 10.0, 20.0))
    first = simulator.simulate(np.ones((8, 8)), 20.0, rng, history)
    second = simulator.simulate(np.zeros((8, 8)), 20.0, rng, history)
    assert first.reads.shape == (3, 8, 8)
    assert second.metadata["persistence_source"] == "detector_history"
    assert np.max(second.science) > 0.0
```

- [ ] **Step 2: Run detector tests and observe the missing simulator failure**

Run: `pytest -q tests/synthetic/test_detectors.py`

- [ ] **Step 3: Implement causal UVIS/IR operations**

UVIS applies photon Poisson noise, dark/bias/read noise, saturation/bleed,
spatial row-dependent CTE trailing, and DQ propagation. IR accumulates
nondestructive reads, applies persistence from prior fluence/saturation,
read-level cosmic rays, nonlinearity, saturation, and a linear ramp fit.
Every effect is separately configurable and labeled.

- [ ] **Step 4: Run focused and existing tests**

Run: `pytest -q tests/synthetic/test_detectors.py tests/synthetic`

- [ ] **Step 5: Commit**

```text
git add src/synthetic tests/synthetic/test_detectors.py
git commit -m "Add causal WFC3 UVIS and IR detector simulations"
```

### Task 5: Parent-conditioned injection and training integration

**Files:**
- Create: `src/synthetic/injection.py`
- Create: `src/synthetic/v2.py`
- Modify: `src/synthetic/__init__.py`
- Modify: `src/training/synthetic.py`
- Test: `tests/synthetic/test_parent_injection.py`
- Test: `tests/training/test_synthetic_stream.py`

**Interfaces:**
- `RealParentInjector(parent, psf_provider).inject(event, population_draw)` modifies parent photons/counts while preserving parent cadence, DQ, uncertainty, and provenance.
- `HubbleSyntheticV2(parent, population_sampler, psf_provider, detector_simulators).generate()` returns paired null/injected observations and a causal layer manifest.
- `iter_parented_synthetic_training_batches(parents, ...)` yields model-ready batches lazily.

- [ ] **Step 1: Write failing parent-injection tests**

```python
def test_parent_injection_changes_only_astrophysical_signal():
    result = HubbleSyntheticV2(make_parent()).generate()
    assert result.null.schedule == result.injected.schedule
    np.testing.assert_array_equal(result.null.dq, result.injected.dq)
    assert result.injected.provenance["parent_observation_id"] == "exp-1"
    assert np.max(np.abs(result.injected.science - result.null.science)) > 0.0

def test_parented_stream_is_lazy_and_bounded():
    batches = iter_parented_synthetic_training_batches([make_parent()], sample_count=1)
    batch = next(batches)
    assert batch.inputs.raster.shape[-2:] == (720, 1280)
```

- [ ] **Step 2: Run focused tests and observe the missing V2 failure**

Run: `pytest -q tests/synthetic/test_parent_injection.py tests/training/test_synthetic_stream.py`

- [ ] **Step 3: Implement parent injection and lazy training conversion**

Use the parent science image and DQ as the null observation. Add only the
astrophysical source perturbation through the selected PSF and detector path;
share the parent noise/history realization between null and injected views.
Keep raw/FLT/IMA/FLC/DRZ stage names in provenance, and provide an optional
subprocess runner hook for `calwf3`/AstroDrizzle without making those external
executables mandatory for unit tests.

- [ ] **Step 4: Run the complete suite and local GPU smoke**

Run: `pytest -q`

Run: `python examples/synthetic_model_smoke.py --device cuda --research`

Require finite loss, successful backward/optimizer step, and RSS/storage
reports within the revised caps.

- [ ] **Step 5: Commit and publish**

```text
git add src/synthetic src/training tests/synthetic tests/training
git commit -m "Add parent-conditioned Hubble synthetic V2"
git push origin main
```

### Task 6: Documentation and realism audit

**Files:**
- Modify: `README.md`
- Modify: `SPEC.md`
- Modify: `docs/SYNTHETIC_DATA.md`
- Modify: `docs/LOCAL_GPU.md`
- Create: `docs/HUBBLE_SYNTHETIC_V2.md`
- Test: `tests/test_public_contract.py`

- [ ] **Step 1: Add the R0–R5 capability matrix**

Document which layers are implemented, which require external empirical
assets, and which require STScI pipeline executables. Explicitly distinguish
`physics_approximation`, `empirical_replay`, `calibrated_approximation`, and
`OOD`.

- [ ] **Step 2: Add real-versus-synthetic audit instructions**

Document a held-out real-parent split by source/system and observation lineage,
plus a real/synthetic discriminator and layer-ablation evaluation. Do not
claim that passing a discriminator test proves scientific validity.

- [ ] **Step 3: Run repository-wide verification**

Run: `pytest -q; python -m compileall -q src tests examples; git diff --check`

- [ ] **Step 4: Commit and push documentation**

```text
git add README.md SPEC.md docs
git commit -m "Document Hubble synthetic V2 realism ladder"
git push origin main
```

## Self-review and explicit boundary

This plan covers the highest-value in-repository upgrade from R0 toward R4.
It does not pretend that empirical PSF retrieval, CRDS reference files,
`calwf3`, AstroDrizzle, or a validated full relativistic/optical model exist
locally. R5 requires installing and version-pinning those external assets and
executables, replaying real Hubble parent files, and validating the exact
calibration chain against real products before it can be claimed.
