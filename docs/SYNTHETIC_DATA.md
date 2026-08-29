# Synthetic data design for Planets-Hubble

**Status:** research/design report
**Date:** 2026-08-30
**Scope:** training and evaluation data for the Hubble-first, wavelength-aware, irregular-cadence multimodal model described in `SPEC.md`.

## Executive recommendation

Use a **hybrid scene-and-injection simulator**. Build the observing schedule,
detector configuration, source field, object context, and nuisance processes from
real archival observations; generate only the latent astrophysical signals that
are difficult to label in the archive; then render and inject those signals into
real noise/systematics whenever possible. Keep a fully synthetic path for unit
tests and rare microlensing cases, but do not use it as the sole source of
training negatives or detector behavior.

The minimum useful simulator is a deterministic, versioned pipeline:

```text
physical scene + source/object posteriors
        -> high-resolution latent flux and surface-brightness fields
        -> transit/eclipse, variability, or microlensing signal
        -> wavelength integration through the actual response curve
        -> exposure-time integration at the actual timestamps
        -> PSF/WCS/pixel sampling and detector effects
        -> Poisson/read/background noise and quality masks
        -> normalized tensors, truth labels, and provenance
```

The archive’s real timestamps, exposure durations, passbands, WCS, detector
metadata, uncertainties, data-quality flags, and neighboring-object fields are
part of the example. They are not optional decoration. A synthetic event that
is sampled on an ideal grid, or injected after a pipeline has removed the
systematic it is meant to test, produces an optimistic recovery estimate.

The first release should target three related but separately scored hypotheses:

1. **Transit/eclipse-like event:** occultation of a luminous source, including
   diluted eclipsing-binary negatives and secondary eclipses.
2. **Stellar microlensing:** a foreground lens magnifies an unrelated
   background source.
3. **Planetary microlensing perturbation:** a planet bound to the foreground
   lens makes a short perturbation to the stellar microlensing event.

The public candidate score may combine these only at a later decision layer.
The simulator and labels must preserve their different causal graphs. In
particular, a microlensing planet is not transiting the background source, and
the foreground lens is not generally the same object as the lensed source.

## Evidence boundary

The following boundary should be maintained in code, metadata, and the eventual
reporting UI.

| Area | Established physics or observation | Proposed approximation for v0 | Open risk or required guard |
| --- | --- | --- | --- |
| Transit/occultation | A limb-darkened source occulted by a moving disk produces an analytic light curve. Mandel & Agol give exact quadratic/nonlinear limb-darkening formulae; `starry` extends analytic spherical-harmonic maps to transits, eclipses, and phase curves. | Use analytic circular-body occultation for the main path; use numerical surface integration only for spots, gravity darkening, asymmetric bodies, or validation. | Limb-darkening coefficients and stellar heterogeneity are correlated with radius ratio and impact parameter. Do not treat a fitted shape as an exact physical measurement. |
| Exposure | The detector measures a time average over the exposure, not an instantaneous sample. | Integrate each latent signal over 16-point Gauss-Legendre or adaptive quadrature, increasing the order around ingress/egress and caustics. | A long exposure can erase an event or shift its apparent midpoint. A midpoint-only simulator will bias duration and timing. |
| Stellar variability | Rotation, spots, evolving active regions, pulsation, granulation, and flares produce non-white variability. Kepler observations show amplitudes from below (10^{-4}) to above 10% in a heterogeneous population. | Combine a surface-map/spot component, a low-frequency correlated component, and a flare point process; calibrate hyperparameters by stellar type and cadence from real light curves. | A single quasi-periodic Gaussian process can look physically interpretable while failing to reproduce spot evolution, flares, or wavelength dependence. |
| Wavelength | Detected counts depend on source spectrum multiplied by the full end-to-end throughput and detector response. WFC3 responses include telescope optics, instrument optics, filter, and detector QE. | Use PHOENIX or another declared stellar spectral library plus a low-dimensional planetary transmission/emission perturbation; convolve with the actual response table. | Atmosphere grids, stellar contamination, filter calibration, and unresolved blends can dominate a shallow color-dependent signal. |
| Detector | WFC3 documentation describes Poisson count statistics, read/dark behavior, bad pixels, nonlinearity, time-variable background, and IR persistence. | Replay empirical nuisance templates from real observations and add calibrated parametric residuals only where data are sparse. | If the same reduction removes a ramp before injection, the model will learn a cleaner distribution than inference receives. |
| Cadence | Astronomical time series are often irregular and windowed. Period searches on uneven samples have different aliasing and false-alarm behavior than regular-grid searches. | Replay real exposure windows exactly; generate schedule variants by block bootstrap of real visit/orbit patterns and explicitly retain gaps. | Randomly interpolating across long gaps leaks an assumed event phase and makes period recovery unrealistically easy. |
| Timing/geometry | A time standard and observer location affect event timing. BJD\_TDB is a practical common event time, and barycentric corrections require the observer and target geometry. | Store UTC/TAI/TT/TDB metadata and use BJD\_TDB for cross-observation event labels; obtain states from a declared ephemeris source. | Mixing header times, mid-exposure times, and barycentric corrections can create artificial transit-timing variations. |
| Stellar microlensing | The point-lens light curve is governed by the lens-source separation in Einstein-radius units. Finite source, parallax, blending, and binary lenses modify it. | Use the analytic point-lens model for pretraining and a finite-source binary-lens solver or validated magnification-map interpolation for planetary perturbations. | Mass, distance, proper motion, and parallax are degenerate; ordinary binary-source events can mimic planetary perturbations. |
| Planetary microlensing | A planet near the source trajectory/stellar Einstein ring creates a short perturbation; the primary observables are mass ratio and projected separation, not a complete orbit. | Draw (q), projected separation (s), trajectory angle, finite-source radius, and (t_E), and label only those parameters that the simulated coverage actually constrains. | Rare-event selection and high-cadence follow-up create severe selection bias. Hubble archival data may be unsuitable for a representative microlensing population. |

Primary references supporting this boundary are [Mandel & Agol 2002,
doi:10.1086/345520](https://doi.org/10.1086/345520), [Luger et al. 2019,
doi:10.3847/1538-3881/aae8e5](https://doi.org/10.3847/1538-3881/aae8e5), the
[STScI WFC3 Data Handbook](https://hst-docs.stsci.edu/wfc3dhb), the [STScI
WFC3 throughput documentation](https://hst-docs.stsci.edu/wfc3ihb/appendix-a-wfc3-filter-throughputs/a-1-introduction),
[Basri et al. 2011, doi:10.1088/0004-6256/141/1/20](https://doi.org/10.1088/0004-6256/141/1/20),
[Eastman et al. 2010, doi:10.1086/655938](https://doi.org/10.1086/655938),
[Paczynski 1986, doi:10.1086/164140](https://doi.org/10.1086/164140), and
[Gaudi 2012, doi:10.1146/annurev-astro-081811-125518](https://doi.org/10.1146/annurev-astro-081811-125518).

## 1. Dataset unit and provenance contract

The atomic training example is an **observation bundle**, not a single image:

```text
bundle = {
  scene_id, source_id, visit_id, exposure_ids,
  t_start, t_mid, t_end, time_scale, barycentric_time,
  image_or_spectrum, uncertainty, data_quality, validity,
  wavelength_or_energy, bandpass_width, response_curve,
  WCS, PSF_or_seeing, pointing, orientation,
  observer_position, observer_velocity,
  object_tokens, coverage_features, provenance,
  latent_truth, observed_label
}
```

Each bundle should retain two distinct provenance layers:

* **Observation provenance:** MAST observation/product IDs, calibration level,
  file hashes, FITS header fields used, WCS solution, response-file version,
  data-quality masks, reduction version, and whether the sample is raw,
  calibrated, drizzled, extracted, or a high-level time series.
* **Simulation provenance:** simulator version, random seed, parameter draw,
  parent real bundle, injection location, injection amplitude, signal component
  IDs, numerical integration tolerance, and post-injection processing path.

The NASA Exoplanet Archive should supply priors and reference provenance, not
unqualified truth. Its Planetary Systems table contains multiple literature
solutions per planet, while the composite table combines values for a more
filled-in statistical view; both expose uncertainties and reference links in
their [column definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html).
Use the solution reference and uncertainty fields when drawing a known system.
Do not silently combine a period from one paper with a radius ratio from an
incompatible solution.

For object context, use Gaia positions, parallax, proper motion, radial
velocity, brightness, and variability when available. The [Gaia ESA
Archive](https://gea.esac.esa.int/archive/) documents these as catalog products,
but catalog absence is not evidence of object absence. Store `catalog_present`,
`catalog_quality`, and `source_detected_in_frame` separately.

## 2. Staged simulator plan

### Stage 0: real-bundle calibration and null corpus

Before making positive injections, assemble a stratified corpus of real HST
bundles:

* WFC3/UVIS imaging, WFC3/IR imaging, and WFC3 grism/time-series products in
  the actual modes selected for the first experiment;
* bright, medium, and faint sources; sparse and crowded fields; different
  detector positions; and multiple visit/orbit patterns;
* real source-free pixels, stable stars, variable stars, known eclipsing
  binaries, and observations with flagged persistence, cosmic rays, or
  incomplete coverage;
* raw/calibrated products where feasible, plus the exact reduction products
  intended for model input.

The null corpus must be split by sky region and source before any injection.
Measure distributions of cadence, exposure duration, count rate, background,
uncertainty, flag rates, PSF/FWHM, WCS drift, pointing, detector position,
temperature/thermal proxies when present, and wavelength coverage. These are
the empirical bases for later parameter draws. A proposed default should be
replaced by an empirical quantile once a sufficiently large corpus exists.

### Stage 1: physical scene and regional object context

Create a tangent-plane scene with a target source and neighboring objects.
Start from real WCS/source detections whenever an injection is attached to a
real observation. For fully synthetic examples, draw a field from a measured
source-density mixture conditioned on Galactic latitude, magnitude, and crowding.

For each object, retain:

```text
position and angular uncertainty
proper motion and uncertainty
parallax/distance posterior, when available
radial velocity, when available
stellar type, Teff, log g, metallicity, extinction posterior
band-integrated brightness and uncertainty
shape/extent, PSF-convolved morphology, and source-detection confidence
```

Generate a **truth object graph** and one or more **observed catalog views**.
The views may omit objects, perturb positions, merge neighbors, or assign an
uncertain foreground/background order. This allows catalog-shortcut tests
without changing the physical scene.

### Stage 2: orbital transit, eclipse, and blend signals

For a target source, draw a Keplerian orbit, calculate sky-plane separation as
a function of time, and evaluate the occultation against a wavelength-dependent
stellar surface brightness. The baseline transit model is:

```text
F_lambda(t) = integral over visible stellar disk of I_lambda(x,y)
              minus the occulted region
```

Use analytic Mandel–Agol/`starry` evaluation for a spherical star and circular
planet. For a secondary eclipse, render the planet’s emitted/reflected map and
remove the visible planet contribution while it is behind the star. For
validation-only surface realism, add spots or faculae to the stellar map and
use numerical integration or a surface-map model.

Always evaluate the signal at sub-exposure times and integrate:

```text
F_exposure = (1 / (t_end - t_start))
             * integral[t_start,t_end] F(t) dt
```

For a real exposure, inject the difference between the with-signal and
without-signal flux at the detector’s native stage. This is safer than adding
a normalized dip after aperture extraction because dilution, background,
nonlinearity, and extraction weights then act on the signal correctly.

Include these astrophysical event classes:

* planet transit, including grazing and single-transit cases;
* planetary secondary eclipse and phase-curve variation where the passband is
  physically sensitive;
* multi-planet sequences with transit-timing offsets;
* bound eclipsing binaries, hierarchical triples, and diluted background
  eclipsing binaries;
* spot-crossing and asymmetric transit shapes as hard transit-like examples;
* no event, but with an orbit prior present in metadata only.

The last class prevents the model from treating a catalog prior as measured
evidence.

### Stage 3: stellar variability and transient activity

Use a mixture, with component identity recorded per example:

1. **Rotational surface component.** Place spots/faculae at random latitude,
   longitude, area, contrast, and lifetime; advect them with rotation and
   optional differential rotation. Compute the wavelength-dependent disk
   integral. This is the preferred physically interpretable component.
2. **Correlated residual component.** Add a low-frequency Gaussian process or
   autoregressive process conditioned on stellar class and cadence to cover
   granulation, pulsation, unresolved activity, and model mismatch. A
   quasi-periodic kernel may be used as an approximation, but its hyperparameters
   must not be labeled as spot latitude, size, or lifetime without validation.
3. **Flares.** Draw event times from a Poisson or clustered point process and
   energies from a truncated power law calibrated by spectral type and activity.
   Use an empirical white-light flare template with correlated amplitude,
   rise time, decay time, and complexity. Map the flare spectrum separately;
   do not assume a wavelength-independent multiplicative impulse.
4. **Pulsation/oscillation.** Add one or more modes with amplitude, frequency,
   phase, and coherence time for relevant stellar populations.

Kepler evidence motivates this mixture: [Basri et al. 2011](https://doi.org/10.1088/0004-6256/141/1/20)
found a broad range of periodic behavior and cautioned that the strongest
period is not always rotation, while [Davenport et al. 2014,
doi:10.1088/0004-637X/797/2/122](https://doi.org/10.1088/0004-637X/797/2/122)
provides an empirical flare morphology/energy basis for active M dwarfs.

### Stage 4: wavelength and spectral response

Represent a source spectrum at sufficiently fine native wavelength resolution,
then calculate detected photons for response (R_b(\lambda)):

```text
mu_b(t) = A_tel * exposure_time
          * integral [ f_lambda(t) * R_b(lambda) * lambda / (h*c) ] d_lambda
```

The exact unit convention must be declared and tested. For WFC3, use the
versioned throughput tables rather than pivot wavelength alone. STScI states
that integrated throughput includes telescope optics, instrument optics,
filter transmission, and detector QE; the [WFC3 IR spectral-element
documentation](https://hst-docs.stsci.edu/wfc3ihb/chapter-7-ir-imaging-with-wfc3/7-5-ir-spectral-elements)
also documents the broad/medium/narrow filters and G102/G141 grisms.

Recommended hierarchy:

* **Host stars:** draw a PHOENIX spectrum or a declared alternative using
  (T_{eff}, \log g, [Fe/H]), and extinction. [Husser et al. 2013,
  doi:10.1051/0004-6361/201219058](https://doi.org/10.1051/0004-6361/201219058)
  describes a broad PHOENIX synthetic-spectra library.
* **Transit:** use the stellar spectrum times a radius-vs-wavelength or
  transmission-height perturbation. Start with a gray radius plus smooth
  Rayleigh-like slope and a small number of broad molecular bands; later add
  retrieval-grade atmosphere models.
* **Eclipse:** use a planet brightness-temperature/albedo approximation tied
  to orbital separation and stellar irradiation, with a gray fallback. Do not
  claim molecular identifications from this stage.
* **Activity:** allow spot/facula/flaring spectra to differ from the photosphere.
* **Blends:** sum every source spectrum before detector sampling. A nearby
  eclipsing binary can therefore produce a diluted, color-dependent apparent
  transit.

For sparse photometry, emit one token per measured band. For grism spectra,
emit native bins plus response/coverage masks and optionally a canonical
log-wavelength representation. Never encode an unmeasured wavelength bin as a
valid zero.

### Stage 5: cadence, observer geometry, and coverage

The default schedule is the exact sequence of real exposures in the parent
bundle. Preserve `t_start`, `t_mid`, `t_end`, time scale, gap lengths, and
visit/orbit boundaries. If a fixed local tensor is required by the model,
interpolate only within a declared continuous window and set an interpolation
mask on every derived value.

Generate schedule variants by resampling complete visit/orbit blocks, not
individual timestamps. Include:

* no-ingress, ingress-only, flat-bottom-only, and ingress-plus-egress coverage;
* one-event and repeated-event baselines;
* long gaps, clustered cadence, and mixed instruments;
* dropped exposures, flagged exposures, and missing wavelength channels;
* exposure durations that are short, comparable to, and longer than ingress.

For each exposure, derive or load:

```text
observer position/velocity in a declared frame
source line of sight, boresight, roll, off-axis angle
solar/lunar elongation and visibility flags when available
WCS and detector orientation
time-system conversions and barycentric arrival time
```

Use [Eastman, Siverd & Gaudi 2010](https://doi.org/10.1086/655938) as the
timing convention reference. Use a declared ephemeris implementation; NASA’s
[NAIF SPICE system](https://naif.jpl.nasa.gov/) is an authoritative model for
spacecraft observation geometry, while [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/manual.html)
can provide observer-target vectors and spacecraft/ground observing sites.
For the Hubble-specific pipeline, record whether the observer state came from
an HST-specific source or a proxy ephemeris; do not silently call an Earth
observer state “HST geometry.”

### Stage 6: detector and reduction systematics

Prefer a **real residual library** over a universal noise formula. For each
instrument/mode, create source-free or stable-source residual templates after
freezing the exact reduction path. Sample a template conditionally on detector
position, count rate, exposure sequence, visit/orbit phase, and background.

Then add only the effects that are represented in the input stage:

* photon counting noise and background Poisson noise;
* read noise, dark current, bias/offset, flat-field residuals, and bad pixels;
* cosmic rays and persistence of flags/charge trails;
* saturation, nonlinearity, count-rate nonlinearity, and time-varying background;
* PSF/trace motion, focus drift, pointing jitter, WCS distortion, aperture
  losses, scan-rate variation, and spectral wavelength shifts;
* orbit/visit ramps or other repeatable detector-history effects;
* correlated residuals represented by a GP or low-rank template only after
  empirical calibration.

The [WFC3 Data Handbook](https://hst-docs.stsci.edu/wfc3dhb) is the source of
truth for the selected WFC3 modes. Its [persistence chapter](https://hst-docs.stsci.edu/wfc3dhb/chapter-8-persistence-in-wfc3-ir/8-1-persistence-in-wfc3-ir)
describes fluence- and history-dependent IR afterglow; its detector chapters
cover nonlinearity, backgrounds, bad regions, and photometric errors. A
practical v0 persistence model may be a decaying history kernel conditioned on
prior pixel fluence, but it must be tagged as an approximation and sampled
from real history sequences when available.

**Injection location rule:** maintain both of these modes and label them:

* `inject_native`: inject latent flux before detector and extraction effects;
* `inject_product`: inject into an already reduced product for fast scale tests.

Only `inject_native` is valid for end-to-end detector-systematics claims.
`inject_product` is useful for model debugging but must not be mixed into the
same recovery estimate without an explicit mode indicator.

### Stage 7: stellar and planetary microlensing

#### Point lens

For a foreground lens and unrelated background source, start with the
Paczynski point-lens model:

```text
u(t) = sqrt(u0^2 + ((t - t0) / tE)^2)
A(u) = (u^2 + 2) / (u * sqrt(u^2 + 4))
F_obs(t) = Fs * A(u) + Fb
```

Here (F_s) is source flux and (F_b) is unrelated blend flux. Draw the
source and lens as separate object tokens and preserve their angular
separation, distance posteriors, proper-motion vector, and inferred role.

#### Planetary perturbation

Use a binary-lens model with mass ratio (q=M_p/M_l), projected separation
(s) in stellar Einstein-radius units, trajectory angle \(\alpha\), finite
source radius \(\rho\), and optional parallax. The planet perturbs the smooth
stellar-lens curve when the source trajectory approaches a planetary or
central caustic. The high-accuracy path should use a validated finite-source
binary-lens solver, adaptive contour integration, or magnification maps. A
coarse caustic approximation is acceptable only for pretraining and must carry
`microlensing_solver_tier` in metadata.

The principal public physical targets are:

```text
stellar-lens event: t0, tE, u0, Fs, Fb, parallax if observable
planetary perturbation: q, s, alpha, perturbation midpoint/duration
```

Do not force `period`, `inclination`, or a three-dimensional semimajor axis
onto this branch. NASA’s [Exoplanet Science Strategy](https://science.nasa.gov/wp-content/uploads/2023/05/3a.201809_ExoplanetScienceStrategy.pdf)
states that microlensing measures planet/star mass ratio and instantaneous
projected separation in Einstein-radius units. [Gould & Loeb 1992,
doi:10.1086/171700](https://doi.org/10.1086/171700) establishes the short-lived
planetary perturbation concept, and [Cassan et al. 2010,
doi:10.1051/0004-6361/200913755](https://doi.org/10.1051/0004-6361/200913755)
discusses finite-source/caustic-crossing inference.

Include hard microlensing negatives: point-lens-only events, binary-source
events, stellar binary lenses, variable background sources, and smooth detector
artifacts. [Gaudi 1997](https://arxiv.org/abs/astro-ph/9706268) describes the
binary-source/planetary-perturbation degeneracy; the simulator should generate
it deliberately rather than treating every short anomaly as a planetary label.

## 3. Proposed parameter distributions

The distributions below are **starting proposals**, not measured population
claims. Fit them to the Stage 0 corpus and the selected archive tables before
using them for population-level statements. Use hierarchical draws so related
parameters remain physically coupled.

| Group | v0 proposal | Coupling and notes |
| --- | --- | --- |
| Stellar (T_{eff}, \log g, [Fe/H], A_V) | Stratified empirical draw from Gaia/Exoplanet Archive posteriors; for fully synthetic stars, (T_{eff}=2500\)–12,000 K with explicit cool-star, solar-type, and hot-star strata. | Draw (M_*,R_*,L_*) from an isochrone or catalog posterior; never draw them independently. |
| Stellar brightness | Empirical magnitude/count-rate quantiles from each instrument/mode, oversampling the faint end and the saturation boundary. | Convert through spectrum and response; do not normalize away count-rate effects before detector simulation. |
| Planet radius ratio (R_p/R_*) | Mixture: log-uniform 0.005–0.03, 0.03–0.1, and 0.1–0.25 components, with class weights recorded. | The injection prior is not an occurrence-rate estimate. Include grazing geometry separately. |
| Period (P) | Log-uniform 0.5–100 d for HST local/event training; a long-baseline stratum to 1000 d for single-transit/weak-period labels. | Condition period on available baseline and report `period_observable`; use catalog posterior draws for known systems. |
| Eccentricity (e) | Circular component plus Beta(0.867, 3.03) proposal for the non-circular component, truncated/conditioned on physical transit geometry. | The Beta values come from a high-S/N RV sample in [Kipping 2013, doi:10.1093/mnrasl/slt075](https://doi.org/10.1093/mnrasl/slt075); treat this as a prior proposal, not universal truth. Draw ω uniformly. |
| Orientation | For a population, isotropic (cos i); for targeted transit injections, condition on a chosen impact-parameter mixture, including (b>1-R_p/R_*) grazing cases. | Keep the selection flag so transit-conditioned examples are not used to estimate occurrence. |
| Limb darkening | Interpolate atmosphere-grid coefficients at (T_{eff},\log g,[Fe/H],A_V), then perturb within coefficient uncertainty. | Validate physical intensity positivity and monotonicity. |
| Eclipse depth | Calculate from planet/star spectra where possible; otherwise log-uniform 1 ppm–1% within a declared band and mark `eclipse_approximation=true`. | Do not reuse transit depth or assume achromaticity. |
| Spots/faculae | Number 0–20 per active region; area, contrast, latitude, lifetime, and differential-rotation shear drawn from empirical strata. | Use low-activity and high-activity regimes; correlate activity with (T_{eff}), rotation, and flare rate. |
| Rotation | Empirical period distribution by stellar class; fallback log-uniform 0.2–100 d. | Spot evolution lifetime proposal: 0.5–20 rotation periods. |
| Flares | Point process with a spectral-class-dependent rate; energy proposal (10^{29})–(10^{34}) erg, broken/truncated power law. | Calibrate against the [Kepler flare catalog](https://arxiv.org/abs/1607.03494) and preserve detection incompleteness rather than extrapolating its observed rate blindly. |
| Wavelength | Use actual WFC3 filter/grism response; for future domains, sample response curves from a versioned instrument library. | Randomize missing bands and response perturbations only within documented calibration uncertainty. |
| Cadence | Exact real schedules first; schedule variants from complete visit/orbit block bootstrap. | Do not draw independent timestamps from a uniform interval. |
| Detector nuisance | Empirical conditional templates; fallback lognormal amplitudes fitted to residual quantiles, not universal fixed values. | Include a zero-effect component so the model cannot assume every example has a ramp or persistence event. |
| Point-lens (t_E) | Log-uniform 1–300 d for broad stress testing, then replace with a Galactic-lens prior for bulge fields. | Couple (t_E) to lens mass, distances, and relative transverse velocity for physically sourced examples. |
| Point-lens (u_0) | Uniform in signed (u_0) over ([-1,1]) for event morphology; oversample small (|u_0|) for recovery curves. | Weighting must be recorded; oversampling is not an event-rate claim. |
| Planetary lens (q) | Log-uniform (10^{-6})–(10^{-2}), with dedicated Earth/Neptune/Jupiter-like strata. | Preserve mass-ratio and projected-separation degeneracies in labels. |
| Planetary lens (s) | Log-uniform 0.3–3 Einstein radii for the primary caustic-sensitive set, plus 0.03–30 stress tails. | Include the close/wide (s\leftrightarrow1/s) degeneracy pairs. |
| Microlensing geometry | α uniform on ([0,2\pi)); ρ log-uniform (10^{-4})–0.5 for finite-source stress; parallax off/on strata. | Use adaptive sampling around caustic crossings; retain lens/source distance and proper-motion posteriors. |
| Crowding/blending | Empirical neighbor separation and flux-ratio distribution by field; synthetic tail with one bright neighbor and unresolved blends. | Explicitly include a catalog-missing neighbor view. |

Use an **event-balanced sampler** for training and an **event-rate-weighted
sampler** for deployment-like evaluation. Store both the draw probability and
the desired evaluation weight so reweighting is possible.

## 4. Injection/recovery design

Every injection should be paired with its exact parent:

```text
parent: real bundle with no injected signal
child: same bundle + one or more injected latent components
difference: child - parent at native and normalized representations
truth: complete simulator parameter record
```

The paired design enables direct signal-preservation tests and prevents a
background field from becoming a label. Generate the following recovery grid:

* signal-to-noise: below threshold, near threshold, and clearly detected;
* event coverage: no event, one edge, one full event, and multiple events;
* depth/duration: shallow-to-deep and exposure-smearing regimes;
* wavelength: single band, sparse multi-band, dense grism, and missing-band;
* variability: quiet, rotational, spot-crossing, flare-rich, and pulsating;
* detector: clean, typical, and extreme-but-realistic systematics;
* crowding: isolated, blended, and catalog-incomplete;
* microlensing: stellar-only, planetary anomaly, binary-source mimic, and
  sparse/caustic-missed schedules.

### Splits

Split by **source/system and sky field**, never by adjacent frames or visits
from one source. Hold out complete instrument modes, epochs, crowding strata,
and cadence families for domain-generalization tests. For known planets, put
all observations and all derived windows of one host system in one split.

### Recovery metrics

Report at fixed false-positive rates and in operating-point-free form:

* source localization: angular error, WCS-consistent match rate, and heatmap
  average precision;
* event detection: PR-AUC, recall at fixed field/source FPR, and event-window
  intersection-over-union;
* transit/eclipse parameters: depth/duration/timing coverage, interval
  calibration, CRPS or log score, and bias by SNR/cadence/wavelength;
* period: recovery only in the `period_observable` stratum; report aliases,
  integer-period errors, and a constraint-status confusion matrix;
* microlensing: stellar-event recall, planetary-perturbation recall at fixed
  (q,s), perturbation timing error, and (q/s) posterior calibration;
* nuisance/OOD: artifact AUROC/AUPRC, missing-modality calibration, and OOD
  detection under held-out instrument/cadence combinations;
* completeness: a multidimensional completeness surface over depth, duration,
  count rate, cadence, wavelength coverage, crowding, and systematics level.

Do not report one aggregate accuracy without these strata. An event-balanced
training sampler can make a rare planetary perturbation appear common, and a
random frame split can make the same astrophysical event appear independently
recovered many times.

## 5. Labels and target semantics

Store three labels for every example: `latent_truth`, `observed_label`, and
`available_evidence`. They answer different questions.

### Source-level labels

```text
source_present_in_truth
source_detected_in_observation
source_ra_dec and uncertainty
source_type: star, galaxy, compact object, blend, unknown
foreground/background role, or unknown
catalog_present and catalog_quality
```

### Event-instance labels

Each event is an interval, not just a class:

```text
event_id, parent_component_ids
class: transit, secondary_eclipse, eclipsing_binary,
       stellar_microlensing, planetary_microlensing_perturbation,
       stellar_variability, flare, detector_artifact, none
t_start, t_mid, t_end in BJD_TDB and original time systems
coverage_fraction and exposure-smearing fraction
event_visible_in_data: yes, partial, no
```

Use soft or interval labels when the source is catalog-confirmed but the event
window is predicted from an uncertain ephemeris. A cataloged planet is not a
positive label for an event outside the observed window.

### Parameter labels and constraint status

For each parameter, store the simulator draw or catalog posterior plus a
visibility assessment:

```text
value_or_distribution
lower, upper, or posterior samples
source: simulated, archive posterior, derived, or unavailable
constraint_status: well_constrained, weakly_constrained,
                   prior_dominated, unconstrained
```

For transit branches include depth, duration, ingress/egress, (R_p/R_*),
impact parameter, period, epoch, inclination, and (e\cos\omega,e\sin\omega)
when appropriate. For the lensing branch include (t_0,t_E,u_0,q,s,\alpha,\rho),
blend, and parallax only when the simulated coverage supports them. Never train
a period head to emit a precise period for a one-event bundle and then call the
result observation-constrained.

## 6. Negative examples and confounder curriculum

Negatives should be generated from the same field/cadence/detector distribution
as positives. The core curriculum is:

1. Real null bundles with no known event injection.
2. Real stable stars with photon/read/background noise only.
3. Rotation/spot variability, pulsation, granulation, and flare-only examples.
4. Detector-only examples: ramp, persistence, cosmic ray, bad-pixel cluster,
   pointing/PSF drift, wavelength shift, and background change.
5. Eclipsing binaries: bound, hierarchical, diluted background, grazing, and
   secondary-only configurations. [Díaz et al. 2013,
   doi:10.1051/0004-6361/201321475](https://doi.org/10.1051/0004-6361/201321475)
   is a useful false-positive reference.
6. Transit-like events with chromatic dilution, spot crossings, or asymmetric
   ingress/egress but no planet label.
7. Stellar microlensing with no planet; planetary perturbation with a source
   that is too sparsely sampled to be observable; and binary-source mimics.
8. Object-context traps: known exoplanet host token with no injected event,
   massive foreground object with no lensing, dense catalog with no signal,
   and foreground/background role swaps.
9. Time-shifted and phase-shifted injections where the window is not centered
   on a candidate. These test for event-alignment leakage.

For the first classifier, “negative” should mean negative for the requested
hypothesis, not physically empty. An eclipsing binary is negative for a planet
transit but positive for an eclipse-like event; ordinary stellar microlensing is
negative for a planetary perturbation but positive for a microlensing event.

## 7. Domain randomization without destroying physics

Randomize nuisance variables in a way that preserves their causal structure:

* sample response curves, detector positions, PSF/trace motion, exposure
  durations, and background histories from real instrument strata;
* perturb WCS/pointing within measured uncertainties and apply the same
  transformation to source positions, object tokens, and pixel data;
* vary stellar spectra, extinction, metallicity, activity, and blend flux
  jointly rather than independently;
* vary cadence by whole observing blocks and preserve long gaps;
* drop wavelength modalities and catalog objects with an explicit missingness
  mechanism, not by replacing them with valid zeros;
* vary injection stage (`native` versus `product`) but keep it as a visible
  domain label and score the modes separately;
* include simulator tiers: analytic, numerical, empirical-template, and
  real-parent. A model should not be rewarded for identifying the tier.

Use leave-one-domain-out tests:

```text
train: several instruments/modes/cadence families
test: an unseen combination, not merely a new random seed
```

Run shortcut probes with catalog identifiers, target names, proposal IDs,
filter names, coverage vectors, and object tokens individually masked. A large
performance collapse means the signal branch is not carrying the intended
measurement evidence.

## 8. Validation checks before training

### Physics and numerical checks

* **Transit geometry:** for a uniform stellar disk and small planet, the
  out-of-transit-normalized depth approaches ((R_p/R_*)^2) away from limb
  darkening and dilution. Compare analytic and numerical occultation curves
  over a grid of radius ratio, impact parameter, and limb-darkening values.
* **Ingress/egress:** convergence of exposure-averaged flux as quadrature
  order increases; compare against a high-resolution reference integration.
* **Flux accounting:** integrating source spectrum through a response curve
  must agree with the emitted count-rate convention and units. The same source
  with a narrower response must not produce more photons solely because its
  pivot wavelength was used as a delta function.
* **Noise:** with a fixed expected count rate, sample mean and variance must
  agree with Poisson plus read-noise predictions; masks must not silently turn
  invalid pixels into zeros.
* **Timing:** verify `t_end - t_start = exposure_duration`, correct time-scale
  transforms, and BJD\_TDB shifts for a known source/observer geometry.
* **Point lens:** recover (A(u)), symmetry about (t_0), and the expected
  (t_E) scaling. As (q\to0), the binary-lens curve must converge to the
  point-lens curve away from caustic numerical tolerances.
* **Finite source:** increasing ρ must smooth caustic structure; magnification
  must remain finite for a resolved source.
* **Role preservation:** swapping foreground/background metadata without
  changing the image must change only role-dependent labels, never the measured
  pixels.

### Statistical and leakage checks

* Compare parent/child differences to the intended injected signal before and
  after normalization; baseline normalization must not erase or amplify events
  in a class-dependent way.
* Re-run injections through the full reduction path and compare native-stage
  and product-stage recovery. Any claim about detector robustness requires the
  native-stage result.
* Compare real and synthetic distributions for count rate, uncertainty,
  residual autocorrelation, gap lengths, event-free periodograms, wavelength
  coverage, object density, PSF, flag rate, and visit/orbit phase.
* Check train/validation/test hashes, source IDs, field IDs, parent observation
  IDs, and cadence-block overlap for leakage.
* Audit the simulator seed and parameter record: every tensor must be
  reproducible from immutable inputs and a recorded version.
* Check label prevalence after all filters, not just before rendering. A transit
  may become invisible after exposure integration; label it `latent_positive`
  but `observable=false` rather than counting it as a recoverable positive.

### Acceptance gates for a first dataset release

Do not train the full model until all of the following are true:

1. 100% of examples have a complete provenance record, valid units, declared
   time scale, and a non-null simulator seed.
2. Analytic/numerical transit agreement passes the predeclared flux tolerance
   on a parameter grid, including grazing and long-exposure cases.
3. Count/noise and response-curve tests pass for every selected instrument mode.
4. Parent/child signal differences are recoverable by a transparent matched
   filter or box/physical baseline at the expected SNR ordering.
5. No source, field, or parent-observation leakage exists across splits.
6. Recovery curves are monotonic within uncertainty with injected depth/SNR in
   each cadence and detector stratum; non-monotonic regions are investigated.
7. Catalog masking, position perturbation, and role-swap probes show that
   object context is auxiliary rather than a sufficient label shortcut.
8. Microlensing and transit branches have separate label schemas and metrics,
   and one branch cannot be scored as the other by an implicit class mapping.

## 9. Recommended implementation order and stopping boundaries

### Milestone A: physically checked photometric core

Implement real-schedule replay, one stellar spectrum, one passband, analytic
transit/eclipse, exposure integration, Poisson/read noise, masks, and paired
parent/child examples. Validate against the checks above before adding a model.

### Milestone B: activity and instrument realism

Add the spot/flare mixture, empirical residual templates, wavelength tokens,
blends, catalog views, WCS perturbation, and HST detector-history effects.
Run recovery stratified by activity and detector nuisance.

### Milestone C: multimodal and long-baseline curriculum

Add sparse/dense wavelength mixtures, multiple visits, missing modalities,
observer states, coverage maps, and period constraint labels. Train a simple
matched-filter/BLS-like and transparent classifier baseline before the full
AstroMamba-H design.

The executable pretraining bridge is
`training.iter_synthetic_training_batches`. It generates one full-resolution
720x1280 bundle at a time, alternates null and injected views, converts the
selected view into `AstroMambaHInputs`, and yields a bounded training batch.
It does not retain a dataset-wide array cache. The standalone
`SyntheticGenerator` continues to support smaller rasters for fast physics
tests and stress checks; full resolution is required at the model-training
boundary.

### Milestone D: microlensing auxiliary branch

Add point-lens pretraining, then finite-source binary-lens planetary
perturbations and hard binary-source negatives. Treat the result as an
auxiliary representation until there is adequate real microlensing coverage.

### Stop conditions

Pause expansion when a proposed component cannot be calibrated to either a
primary physics model or a real residual distribution. Mark it as an explicit
OOD/approximation tier instead of filling the gap with arbitrary noise. In
particular, do not claim Hubble archival completeness for planetary microlensing
from a simulator alone, and do not report unconstrained orbital elements as
measurements.

## 9.1 Pretraining, real-image post-training, and holdout policy

Synthetic data is the pretraining domain, not the final evidence domain. The
training lifecycle is:

```text
synthetic curriculum
    -> synthetic validation
    -> frozen real HST post-training/fine-tuning set
    -> real HST held-out evaluation
```

Real images, calibration products, jitter/engineering files, and their hashes
must be retained outside the source repository in the configured data store.
The repository should contain manifests, provenance, and checksums rather than
large FITS/image payloads. The real-image post-training and evaluation split
must be made by source/system and observation lineage, with no parent exposure
or near-duplicate image appearing in synthetic calibration and evaluation.

Synthetic examples must carry a causal layer manifest, for example:

```text
astrophysical_signal
stellar_variability
spectral_response
exposure_integration
optical_psf_and_aberration
pointing_and_jitter
detector_effects
background_and_crowding
timing_and_observer_geometry
noise_realization
```

Each layer can be disabled, randomized, or replayed from a real parent. This
lets us measure whether the model learned a transit rather than a telescope
or a simulator fingerprint.

## 9.2 Instrument-realistic nuisance curriculum

The simulator should use instrument-specific nuisance families and should not
collapse every loss into an undifferentiated Gaussian noise term.

For Hubble/WFC3 and related imaging data, the initial nuisance vocabulary is:

* orbital thermal breathing: focus-dependent PSF width, asymmetry, and
  centroid changes;
* fine-guidance jitter, drift, roll, guide-star acquisition state, and visit
  offsets, including a jitter-file replay path;
* field-dependent optical aberration and geometric distortion, including
  pixel-area-map effects when using flat-fielded detector products;
* UVIS radiation damage and position/morphology-dependent CTE charge loss;
* IR persistence/afterglow after high-fluence exposures and detector
  nonlinearity along the read ramp;
* cosmic-ray hits, hot pixels, bad columns, saturation, background gradients,
  stray light, and shutter-dependent blur where relevant;
* time-dependent sensitivity and amplifier/flat-field residuals.

These are documented HST behaviors, not arbitrary augmentations. The simulator
should draw their amplitudes from instrument/epoch strata or replay measured
residual templates. If no calibration distribution is available, the layer is
marked `approximation` or `OOD` and is not silently treated as HST truth.

For Kepler/K2-like pretraining domains, include the quarterly roll/season
state, channel/module-dependent response, target-pixel and aperture changes,
thermal/pointing trends, cosmic rays, impulsive spikes, and common-mode
systematics. A co-trending-basis-vector-like layer may be used as a nuisance
representation, but it must be labelled as a correction/systematics feature
and not as an astrophysical signal. Kepler-derived examples remain a separate
instrument domain until the HST-to-Kepler transfer behavior is measured.

## 9.3 Timing, relativity, extinction, and three-dimensional context

The phrase “relativistic loss” must be decomposed into physically distinct
effects:

1. Convert observation times through declared UTC/TAI/TT/TDB scales and retain
   exposure start, midpoint, and end. Use barycentric light-travel-time
   corrections for event timing; use JPL ephemerides when their accuracy is
   needed.
2. Retain apparent-position terms from observer motion, including stellar
   aberration and one-way light time. General-relativistic light bending and
   gravitational delay are separate optional terms and must not be implied by
   a Newtonian aberration correction.
3. Apply distance dilution, wavelength-dependent interstellar extinction and
   scattering, telescope throughput, PSF/aperture losses, detector quantum
   efficiency, CTE loss, persistence, and saturation as separate causal layers.
   These are not relativistic effects.
4. Use a three-dimensional scene/context map to provide source and lens
   distances, proper motions, dust/extinction along the line of sight,
   foreground/background ordering probabilities, and mass/redshift priors.
   A 3D map may condition a lensing calculation; it must not be used as a
   shortcut label for whether an event is a planet.

For every generated example, store the correction tier and provenance:

```text
timing_scale: BJD_TDB
observer_state_source: declared ephemeris or synthetic orbit
light_time_model: none | solar_system_barycentric | finite_distance
aberration_model: none | apparent_newtonian | validated_relativistic
extinction_model: none | catalog_map | synthetic_3d_dust
instrument_effect_tier: measured_replay | calibrated_approximation | OOD
```

The implementation should begin with barycentric timing and observer-state
features, then add dust and lens geometry. It should not invent a strong-field
relativity signal for ordinary HST exoplanet fields where the expected effect
is below the photometric/timing error budget.

## 10. References

The links below are the primary or first-party references used for the design.

1. Mandel, K. & Agol, E. (2002), “Analytic Lightcurves for Planetary Transit
   Searches,” *ApJ*. [doi:10.1086/345520](https://doi.org/10.1086/345520).
2. Luger, R. et al. (2019), “STARRY: Analytic Occultation Light Curves,” *AJ*.
   [doi:10.3847/1538-3881/aae8e5](https://doi.org/10.3847/1538-3881/aae8e5).
3. Luger, R. et al. (2022), “Analytic Light Curves in Reflected Light,” *AJ*.
   [doi:10.3847/1538-3881/ac4017](https://doi.org/10.3847/1538-3881/ac4017).
4. Basri, G. et al. (2011), “Photometric variability in Kepler target stars. II,”
   *AJ*. [doi:10.1088/0004-6256/141/1/20](https://doi.org/10.1088/0004-6256/141/1/20).
5. Davenport, J. R. A. et al. (2014), “Kepler Flares II,” *ApJ*.
   [doi:10.1088/0004-637X/797/2/122](https://doi.org/10.1088/0004-637X/797/2/122).
6. Davenport, J. R. A. et al. (2016), “The Kepler Catalog of Stellar Flares,”
   *ApJ*. [arXiv:1607.03494](https://arxiv.org/abs/1607.03494).
7. Gibson, N. P. et al. (2012), “A Gaussian process framework for modelling
   instrumental systematics,” *MNRAS*.
   [doi:10.1111/j.1365-2966.2011.19915.x](https://doi.org/10.1111/j.1365-2966.2011.19915.x).
8. Husser, T.-O. et al. (2013), “A new extensive library of PHOENIX stellar
   atmospheres,” *A&A*. [doi:10.1051/0004-6361/201219058](https://doi.org/10.1051/0004-6361/201219058).
9. VanderPlas, J. T. (2018), “Understanding the Lomb–Scargle Periodogram,”
   *ApJS*. [doi:10.3847/1538-4365/aab766](https://doi.org/10.3847/1538-4365/aab766).
10. Eastman, J., Siverd, R. & Gaudi, B. S. (2010), “Achieving Better Than 1
    Minute Accuracy in the Heliocentric and Barycentric Julian Dates,” *PASP*.
    [doi:10.1086/655938](https://doi.org/10.1086/655938).
11. Paczynski, B. (1986), “Gravitational microlensing by the galactic halo,”
    *ApJ*. [doi:10.1086/164140](https://doi.org/10.1086/164140).
12. Gould, A. & Loeb, A. (1992), “Discovering Planetary Systems through
    Gravitational Microlenses,” *ApJ*.
    [doi:10.1086/171700](https://doi.org/10.1086/171700).
13. Gaudi, B. S. (2012), “Microlensing Surveys for Exoplanets,” *ARA&A*.
    [doi:10.1146/annurev-astro-081811-125518](https://doi.org/10.1146/annurev-astro-081811-125518).
14. Cassan, A. et al. (2010), “Bayesian analysis of caustic-crossing
    microlensing events,” *A&A*. [doi:10.1051/0004-6361/200913755](https://doi.org/10.1051/0004-6361/200913755).
15. Gaudi, B. S. (1997), “Planetary Microlensing Perturbations: True Planets or
    Binary Sources?” [arXiv:astro-ph/9706268](https://arxiv.org/abs/astro-ph/9706268).
16. Kipping, D. M. (2013), “Parametrizing the exoplanet eccentricity
    distribution with the Beta distribution,” *MNRAS Letters*.
    [doi:10.1093/mnrasl/slt075](https://doi.org/10.1093/mnrasl/slt075).
17. Díaz, M. R. et al. (2013), “The contribution of secondary eclipses as
    astrophysical false positives,” *A&A*.
    [doi:10.1051/0004-6361/201321475](https://doi.org/10.1051/0004-6361/201321475).
18. Pagul, A. & Rivera, I. et al. (2024), *WFC3 Data Handbook, Version 6.0*,
    STScI. [Official handbook](https://hst-docs.stsci.edu/wfc3dhb).
19. Marinelli, M. & Green, J. (2025), *WFC3 Instrument Handbook, Version 18.0*,
    STScI. [Official handbook](https://hst-docs.stsci.edu/wfc3ihb).
20. NASA Exoplanet Archive, “Planetary Systems and Planetary Systems Composite
    Parameters Data Column Definitions.” [Official documentation](https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html).
21. NASA, *Exoplanet Science Strategy*, microlensing section.
    [Official strategy PDF](https://science.nasa.gov/wp-content/uploads/2023/05/3a.201809_ExoplanetScienceStrategy.pdf).
22. STScI, *WFC3 Data Handbook*, persistence in WFC3/IR.
    [Official persistence documentation](https://hst-docs.stsci.edu/wfc3dhb/chapter-8-persistence-in-wfc3-ir/8-1-persistence-in-wfc3-ir).
23. STScI, *WFC3 Data Handbook*, UVIS calibration and CTE.
    [Official UVIS calibration documentation](https://hst-docs.stsci.edu/wfc3dhb/chapter-3-wfc3-data-calibration/3-2-uvis-data-calibration-steps).
24. STScI, *WFC3 Instrument Handbook*, optical performance and breathing.
    [Official optical-performance documentation](https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-6-uvis-optical-performance).
25. STScI, *DrizzlePac Handbook*, HST pointing accuracy and stability.
    [Official pointing documentation](https://hst-docs.stsci.edu/drizzpac/chapter-4-astrometric-information-in-the-header/4-4-hst-pointing-accuracy-and-stability).
26. NASA Exoplanet Archive, *Kepler Data Products Overview*.
    [Official Kepler product documentation](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_Data_Products_Overview.html).
27. Astropy, *Time and Dates*, barycentric and heliocentric light-travel-time corrections.
    [Official time documentation](https://docs.astropy.org/en/stable/time/index.html).
28. JPL NAIF, *Aberration Corrections Required Reading*.
    [Official SPICE aberration documentation](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/abcorr.html).
29. ESA Gaia, *Total Galactic Extinction maps*.
    [Official Gaia extinction-map documentation](https://www.cosmos.esa.int/web/gaia/dr3-extinction-as-function-of-l-b).

## Conclusion

The simulator should make the model confront the same ambiguity that an
astronomer confronts: a shallow event is a convolution of astrophysics,
cadence, wavelength response, detector history, crowding, and missingness.
The strongest v0 design is therefore not a larger synthetic catalog. It is a
small, reproducible set of paired real-parent injections with explicit causal
labels, calibrated nuisance distributions, physically integrated exposures,
separate microlensing/transit semantics, and recovery results reported by
coverage and domain.
