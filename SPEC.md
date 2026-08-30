# Planets-Hubble: Multimodal Exoplanet Discovery Specification

Status: design draft; executable bounded implementation is tracked below

Implementation status (2026-08-30): the repository currently provides a
82,541,531-parameter research preset with FPN spatial features, persistent
source anchors, optional normalized `source_xy` hints for intentionally
uncentered patches, wavelength/object/geometry encoders, structured logits,
source-specific heatmaps, period-constraint probabilities, and per-source
transit-time offsets. Synthetic and real-parent workers run on CUDA with a
hard 1.8 GiB process-RSS cap. The proposed 82--86M Mamba-2 target remains a
scaling target: the installed portable temporal backend is still the gated
convolution fallback unless `mamba_ssm` is explicitly available and tested.
The bounded current training probes are integration evidence, not calibrated
exoplanet sensitivity evidence.

## 1. Purpose

Planets-Hubble will investigate whether a multimodal, wavelength-aware,
spatiotemporal model can find exoplanet-like signals in archival astronomical
observations and estimate which orbital and spectral properties are constrained
by the data.

The system is intended to ingest observations from Hubble first, while keeping
the input contract open to other observatories and future instruments. The
initial scientific signal is an unresolved change in a host star's measured
light, especially a transit or eclipse. Hubble only rarely directly resolves
an exoplanet; most Hubble exoplanet measurements use stellar light and
spectroscopy.

Useful background:

- [NASA: Hubble Exoplanets](https://science.nasa.gov/mission/hubble/science/science-behind-the-discoveries/hubble-exoplanets/)
- [NASA: What is a transit?](https://science.nasa.gov/exoplanets/whats-a-transit/)
- [MAST API documentation](https://mast.stsci.edu/api/v0/)
- [NASA Exoplanet Archive API](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html)

This project is a research system. A model score is a candidate-ranking
signal, not an astronomical confirmation. Confirmation requires independent
observations and scientific validation.

## 2. Design principles

1. Preserve physical coordinates. Wavelength, photon energy, time, angular
   scale, exposure duration, and observer geometry are part of the measurement
   rather than arbitrary telescope metadata.
2. Avoid categorical instrument shortcuts. Telescope names, proposal IDs,
   target names, and filter names should not be primary model inputs.
3. Preserve missingness and uncertainty. Missing observations, bad pixels,
   and interpolated values must remain distinguishable from valid measurements.
4. Keep irregular time information. A long-baseline sequence must preserve
   the actual gaps between observations.
5. Separate evidence from prior information. The observation-derived signal,
   catalog priors, coverage quality, and model uncertainty should remain
   separately inspectable.
6. Support variable wavelength coverage. The model must accept RGB-like
   measurements, sparse photometric bands, dense spectra, or mixtures of these.
7. Do not imply unconstrained orbital parameters. The model must be able to
   report that a period or orbital element is not constrained by the available
   observations.

## 3. Data discovery scope

The first data-processing layer will use MAST's REST API.

### 3.1 Named target workflow

1. Resolve a target name with `Mast.Name.Lookup`.
2. Use the returned ICRS coordinates to query HST observations.
3. Use `Mast.Caom.Cone` or `Mast.Caom.Filtered.Position` to search by sky
   position and radius.
4. Restrict results to public science observations and selected instruments,
   product types, calibration levels, and passbands.
5. Expand each MAST product group with `Mast.Caom.Products`.
6. Record product URIs and observation metadata in a manifest before bulk
   downloading data.

### 3.2 Sky-patch workflow

The same observation search will accept an explicit sky position and angular
footprint. A patch is a celestial region, not merely a rectangle of native
detector pixels.

Every selected observation will be reprojected into a common celestial grid
using its astrometric solution/WCS. The model input raster will have a fixed
size of 1280x720, while the physical angular footprint remains variable and is
provided to the model as an embedding.

### 3.3 High-level science products

MAST's HLSP API will be queried separately for already-processed HST products
whose product type includes time series. HLSPs are useful for prototyping and
cross-checking, but their conventions are heterogeneous and they should not be
assumed to form a uniform training set.

### 3.4 Manifest requirements

The discovery manifest must contain, at minimum:

```text
patch identifier
source identifier, when available
target coordinates and sky footprint
MAST observation/product identifiers
download URI
observation start, midpoint, and end time
exposure duration
instrument-independent wavelength/passband information
calibration level
image/spectrum product type
WCS and spatial footprint information
observer position and velocity, when available
pointing and orientation information, when available
coverage and quality summaries
```

The manifest is an index and provenance record. It is not itself a training
label.

## 4. Normalized multimodal input

The core observation representation is a variable-length collection of
measurement tokens rather than a fixed RGB channel array.

### 4.1 Measurement token

Each photometric, spectral, or sensor-band measurement becomes a token:

```text
measurement_token = {
    normalized_value,
    normalized_uncertainty,
    validity_mask,
    interpolation_mask,
    wavelength_or_energy,
    bandpass_width,
    spectral_response,
    exposure_duration,
    exposure_coverage_fraction,
    optional_spatial_position
}
```

The number of tokens can vary per observation. RGB creates three wavelength
tokens; a spectrograph creates many; a sparse radio or gamma-ray observation
creates the tokens that it actually measures.

### 4.2 Wavelength-aware embedding

Wavelength or photon energy will be encoded continuously, preferably on a
logarithmic scale:

```text
wavelength_embedding = E(log10(wavelength))

measurement_embedding = MLP(
    wavelength_embedding,
    normalized_value,
    normalized_uncertainty,
    log10(bandpass_width),
    log10(exposure_duration),
    validity_mask
)
```

The model will pool the available tokens with attention or another permutation
invariant mechanism. The result is a learned hyperspectral representation that
can accept a variable number of wavelength bins.

The architecture is open to any wavelength range, but generalization to a new
domain is not automatic. A model trained only on optical and infrared data
should not be expected to understand radio or gamma-ray signals without
representative training data and appropriate calibration.

### 4.3 Normalization

The preprocessing layer will create two complementary normalized views:

```text
physical-ratio view:
    measured_flux / robust_baseline

noise-scaled residual view:
    (measured_flux - robust_baseline) / robust_noise_scale
```

For spectra, the equivalent views are continuum-normalized flux and
uncertainty-scaled residuals.

Normalization must not use a transit's minimum as the baseline by accident. The
baseline estimator must be robust to outliers and candidate events, and the
normalization method must be recorded in the manifest.

Per-channel normalization must not erase wavelength-dependent transit depth.
The physical-ratio view and the uncertainty-scaled view should therefore be
retained together where data volume permits.

### 4.4 Exposure duration

Exposure duration is both a normalization factor and a temporal instrument
effect.

Measurements should generally be converted to a rate-like quantity before
normalization, but the original duration must remain available to the model.
Finite exposures average the signal over an interval, so simulated training
events must be integrated over the same exposure window:

```text
observed_flux = true_flux convolved with exposure_window
```

The input should retain `t_start`, `t_mid`, `t_end`, and exposure duration.

## 5. Temporal structure

The model uses multiple temporal coordinates:

```text
t_long:
    actual observation times across visits, months, or years

delta_t:
    time since the previous observation or visit

tau_event:
    fixed-step local time relative to a candidate event

phi:
    optional orbital phase under a proposed period
```

The conceptual input is:

```text
X[long_time, short_time, spatial_location, measurement_token]
```

The short-time axis may be interpolated onto fixed steps inside a continuous
local window. Interpolation across long observational gaps is forbidden. Every
interpolated value receives an interpolation mask.

The model must search for event windows rather than receiving only perfectly
centered known transits. Known transit epochs can be used for positive labels,
but training must also include shifted windows and realistic negative windows
so that event alignment is not leaked into the input.

## 6. Sky-patch geometry

Each input raster is 1280x720 but has a physical geometry embedding:

```text
angular_width
angular_height
solid_angle
angular_pixel_scale_x
angular_pixel_scale_y
projection_distortion
```

These values should be represented on stable scales, generally using logarithms
for positive quantities.

The model supports two inference modes:

```text
discovery mode:
    no source coordinates supplied; produce a source/candidate heatmap

query mode:
    optional source x,y or celestial coordinates supplied; evaluate those
    sources while retaining the surrounding patch as context
```

During training, provided source coordinates should sometimes be dropped so
the model does not become dependent on an external source list.

## 7. Observatory and observer geometry

The model may combine observations from Hubble, spacecraft, ground
observatories, and future platforms through physical geometry rather than
telescope identity.

### 7.1 Observer state

The preferred observer-state vector is:

```text
r_geo       observer position relative to Earth
v_geo       observer velocity relative to Earth
r_bary      observer position relative to the solar-system barycenter
v_bary      observer velocity relative to the solar-system barycenter
```

These vectors must use common reference frames and consistent units. Ground
observatory states can be derived from the observatory location and time;
spacecraft states can be obtained from an orbit or ephemeris.

### 7.2 Pointing and orientation

The observation geometry should include:

```text
boresight unit vector
patch/source line-of-sight unit vector
camera roll or orientation
angular separation from boresight
field rotation rate, when available
```

Angles should be represented with sine/cosine pairs where wraparound matters.
For example:

```text
off_axis_angle = arccos(boresight · source_direction)
roll_embedding = [sin(roll), cos(roll)]
```

Other derived angles may be included when available:

```text
solar elongation
lunar elongation
observer zenith angle
parallactic angle
ecliptic latitude
```

These quantities help with registration, visibility, and systematics. They are
not themselves evidence for an exoplanet.

### 7.3 Combining observatories

Observer position alone does not register images. Cross-observatory pooling
requires WCS reprojection, timestamps on a consistent time scale, pointing
orientation, pixel scale, PSF/seeing characterization, passband information,
and uncertainty propagation.

Independent measurements can improve detection of a faint dip when they overlap
the same physical event and have sufficiently independent noise. Observations
of the same sky region months apart do not automatically create a higher-SNR
measurement of one transit.

## 8. Coverage representation

Both a patch-level coverage vector and a spatial coverage map will be used.

### 8.1 Patch-level coverage vector

```text
total exposure time
valid observation count
exposure count by wavelength regime
longest temporal baseline
median cadence
cadence distribution
missing-data fraction
estimated noise floor
limiting sensitivity
median angular resolution
```

### 8.2 Spatial coverage map

```text
coverage_map[y, x, feature]
```

Possible map features include:

```text
total exposure time at this position
number of visits
wavelength coverage
median uncertainty
median PSF/seeing width
temporal baseline
cadence or visit density
```

Coverage affects detectability and uncertainty, so it must be available to the
quality and active-learning branches. It must not become a shortcut for the
planet label. The system should report performance separately for high- and
low-coverage regions and should support coverage dropout during training.

## 9. Regional object context and gravitational lensing

Every patch embedding will include a representation of the other astronomical
objects present within or near the patch. This context is important both for
source deblending and for a future gravitational-lensing branch.

### 9.1 Object tokens

Objects may come from a catalog, a coadd/source-detection pipeline, or the
model's own spatial detector. The object context should not depend on object
names as primary identifiers. Each object is represented by physical and
spatial features:

```text
object_token = {
    relative_tangent_plane_x,
    relative_tangent_plane_y,
    angular_separation,
    angular_extent,
    shape_or_morphology,
    multi-band_normalized_brightness,
    wavelength_coverage,
    parallax_or_distance_prior,
    proper_motion,
    radial_velocity, when available,
    mass_or_lens_mass_prior, when available,
    redshift, when available,
    astrometric_uncertainty,
    source_detection_confidence
}
```

The object list should include more than known exoplanet hosts. It should also
include likely foreground stars, background stars, galaxies, galaxy clusters,
compact objects, and nearby sources that may contaminate a light curve.

The phrase "larger objects" should be interpreted primarily as objects with
greater mass or a stronger lensing geometry. Apparent angular size alone is not
a reliable proxy for lensing strength. A compact foreground star can be a more
relevant lens than a large-looking diffuse source.

### 9.2 Object-context fusion

The object tokens should be processed by a spatial set encoder or object graph
encoder and fused into every relevant frame/event representation:

```text
regional object tokens
        ↓
spatial set/graph encoder
        ↓
regional context embedding
        ↓
cross-attention with image, spectral, and temporal tokens
```

The context embedding should be available at three levels:

```text
patch level:
    all objects and their geometry in the 1280x720 region

source level:
    neighboring objects around a candidate source

frame level:
    objects present, missing, or changing in that observation
```

The object context should be recomputed or quality-checked after WCS
reprojection. A catalog entry is not automatically a valid source in every
frame, and catalog incompleteness must be represented explicitly.

### 9.3 Lensing geometry

The lensing branch should model the relationship between:

```text
foreground lens object
background source object
observer
relative proper motion
mass and distance priors
time-dependent angular alignment
```

Useful derived features include:

```text
lens-source angular separation
lens-source relative proper motion
observer-source-lens geometry
estimated Einstein angular scale
normalized impact parameter
estimated Einstein crossing time
```

The model should receive the physical inputs needed to estimate these values,
but should also be able to work when mass or distance is uncertain. The output
must expose those uncertainties rather than silently filling missing values.

### 9.4 Two lensing regimes

The system should distinguish two related but different tasks.

Strong or macrolensing by a massive foreground object or structure may produce
arcs, multiple images, or extended distortions. This is primarily a spatial
image/context problem.

Microlensing by a foreground star is primarily a time-dependent brightness
problem. A planet orbiting the foreground lens can produce a shorter, smaller
perturbation in the stellar microlensing light curve. NASA describes this
planetary perturbation as a short additional signal whose timing and amplitude
can constrain the planetary system. [NASA: Hubble's Gravitational Lenses](https://science.nasa.gov/mission/hubble/science/universe-uncovered/hubbles-gravitational-lenses/),
[NASA: How We Find and Characterize Exoplanets](https://science.nasa.gov/exoplanets/how-we-find-and-characterize/)

The model should therefore have separate hypotheses for:

```text
ordinary stellar microlensing
planetary microlensing perturbation
transit or eclipse
stellar variability
instrumental or reduction artifact
```

The planetary-lensing branch must not assume that the foreground lens is the
same star as the background source. This foreground/background distinction is
central to the physics.

### 9.5 Internal microlensing representation

Microlensing is an internal representation and auxiliary training objective in
the first public output design. It should help the shared encoder understand
foreground/background relationships, mass-dependent perturbations, and
time-dependent alignment without forcing every normal candidate result to
expose a lensing classification.

The internal lensing branch may estimate:

```text
microlensing_probability
planetary_perturbation_probability
foreground_lens candidate position
background_source candidate position
event midpoint posterior
Einstein timescale posterior
impact parameter posterior
mass-ratio posterior
projected separation posterior
relative proper-motion posterior
astrometric-separation or centroid-shift prediction
```

These are not the same as the transit-orbit outputs. Microlensing commonly
constrains a planet's mass ratio and projected separation during a transient
event, not a complete orbital period and three-dimensional orbit. A full orbit
should be reported only when additional observations constrain it.

The primary v0 result should expose only the shared candidate/event outputs.
Internal lensing states may be retained as diagnostic tensors or auxiliary
losses for research, but they are not required in the public candidate schema.

### 9.6 Object-context safeguards

Object catalogs and coverage maps can create severe label shortcuts. The model
must not conclude that a patch contains a planet merely because it contains a
known massive star, a cataloged exoplanet host, or a dense object catalog.

Required safeguards include:

```text
catalog-object dropout
object-position perturbation tests
blind tests with target names and planet identifiers removed
separate evaluation on catalog-complete and catalog-incomplete regions
foreground/background role swaps in negative examples
synthetic lensing injections with non-lensing object fields
```

Object context should improve physical interpretation and source association;
the measured time-varying signal must remain the evidence for the candidate.

## 10. Proposed model architecture

```text
                         RAW OBSERVATION
             1280x720 + uncertainty / masks
                              │
                              ▼
                    ┌─────────────────────┐
                    │ MULTISCALE CNN      │
                    │ ConvNeXt-V2 style   │
                    │ approximately 27M   │
                    └─────────┬───────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
           /8 map           /16 map           /32 map
        160x90xC          80x45xC           40x23xC
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                   SOURCE-AWARE TOKENIZER
                deformable ROI / learned top-K
                         approximately 3M
                              │
                    64-192 source/context tokens
                         approximately 384-d
                              │
             ┌────────────────┼─────────────────┐
             ▼                ▼                 ▼
       wavelength        object/context     observer/geometry
         encoder          set/graph encoder       MLP
          ~4M                 ~5M                 ~1M
             │                │                 │
             └────────────┬───┴────────────┬────┘
                          ▼                ▼
                  CROSS-MODAL FUSION
                       approximately 5-6M
                          │
                          ▼
                    LOCAL MAMBA-2
                       approximately 11-13M
                          │
                     EVENT TOKENS
                          │
                          ▼
                    LONG-TIME MAMBA-2
                       approximately 10-12M
                          │
             ┌────────────┴─────────────┐
             ▼                          ▼
       SPATIAL-TEMPORAL              GLOBAL HEADS
          DECODER                       approximately 3M
          approximately 9-10M             │
             │                            ├─ candidate probability
             ▼                            ├─ event probability
      wavelength heatmaps                 ├─ transit time
      source heatmaps                     ├─ depth and duration
      uncertainty maps                    ├─ period posterior
      segmentation                        ├─ parameter uncertainty
                                            ├─ artifact probability
                                            └─ OOD / quality
```

The full 1280x720 patch remains the input. Internally, a multiscale spatial
encoder may first detect sources at coarse resolution and then extract
high-resolution features around candidate sources.

The representation decoder is the primary learned output stage. The event
classifier is intentionally simple in the first system so that we can measure
whether the learned heatmaps contain useful astrophysical information without
also hiding the entire decision inside a second large neural network.

The internal lensing representation can participate in shared encoding and
auxiliary training losses, but the v0 classifier should primarily read the
dense measurement-derived heatmaps, uncertainty, and validity masks.

The target size for this ambitious architecture is approximately 82-86M
parameters, with an allowed research range of roughly 78-90M. Capacity should
go into source-aware spatial modeling, wavelength fusion, object-context
reasoning, temporal hierarchy, and decoding rather than only making Mamba
deeper.

### AstroMamba-H components

#### Spatial backbone: approximately 27M parameters

The spatial backbone is a custom ConvNeXt-V2-style CNN. An initial target
configuration is:

| Stage | Resolution | Channels | Blocks |
| --- | ---: | ---: | ---: |
| Stem | 320x180 | 96 | - |
| Stage 1 | 320x180 | 96 | 3 |
| Stage 2 | 160x90 | 192 | 3 |
| Stage 3 | 80x45 | 384 | 9 |
| Stage 4 | 40x23 | 512 | 3 |

It directly receives normalized physical-ratio flux, noise-scaled residual,
uncertainty, validity mask, interpolation mask, and exposure/coverage maps.
Large depthwise kernels, deformable convolution in the final spatial stages,
anti-aliased downsampling, coordinate channels, PSF conditioning, and FPN
lateral connections are architecture variants to evaluate.

#### Source-aware tokenizer: approximately 3-3.5M parameters

The CNN feature maps must not be globally averaged. An FPN source-proposal head
will produce objectness/transientness maps, select approximately 64-128 top-K
locations, and use deformable multi-scale ROI sampling to create approximately
384-dimensional source tokens. Approximately 32-64 background/context tokens
should be retained alongside the source tokens.

#### Wavelength encoder: approximately 4M parameters

The continuous wavelength encoder uses Fourier features over logarithmic
wavelength or photon energy and accepts a variable number of wavelength bins.
It produces approximately 256-dimensional wavelength tokens from normalized
value, uncertainty, wavelength, bandwidth, exposure duration, response, and
validity. Very dense spectra may first be locally compressed with a small 1D
CNN or Mamba.

#### Object/context encoder: approximately 4.5-5M parameters

Regional objects are processed as a graph or set with a Graph Transformer or
Set Transformer. This branch produces per-object embeddings, a regional
context token, and foreground/background relationship features. It models
source contamination, relative proper motion, mass/distance relationships,
and possible lensing geometry.

#### Observer/geometry encoder: approximately 1M parameters

Observer position and velocity, pointing, angular scale, exposure duration,
orientation, and coverage are encoded with a small MLP and feature-wise
conditioning. Physical vectors and angle sine/cosine pairs are preferred over
categorical observatory IDs.

#### Cross-modal fusion: approximately 5-6M parameters

Limited cross-attention is used where explicit association is needed:

```text
source token
      ├── cross-attention to wavelength tokens
      ├── cross-attention to object-context tokens
      └── FiLM/conditioning from geometry, exposure, and coverage
```

The initial target is three to four cross-modal blocks. The design principle is
CNN for space, attention for modality association, and Mamba for time.

#### Local/event Mamba-2: approximately 11-13M parameters

The first temporal stage operates on source tokens within each visit or
continuous observing sequence. An initial target is width 512, approximately
eight Mamba-2 blocks, and state size 64-128. It receives event-relative time,
delta-t, exposure duration, start/mid/end times, barycentric timing, and
validity/interpolation masks.

It learns local transit shape, ingress, egress, asymmetry, stellar flares,
pointing jitter, cosmic-ray behavior, short-timescale microlensing, and
wavelength-dependent changes.

#### Visit/event pooling

Local sequences are compressed with learned attention pooling into a small set
of event morphology, event photometry, event spectral, and quality tokens.
Blind averaging is not sufficient because it can erase short events.

#### Long-time Mamba-2: approximately 10-12M parameters

The second temporal hierarchy operates over visit/event tokens spanning months
or years with irregular cadence. An initial target is width 512 and seven to
eight Mamba-2 blocks. It receives absolute observation time, delta-t,
long-baseline duration, event confidence, coverage, wavelength availability,
and observer geometry.

It learns recurrence, approximate period, phase consistency, repeated dip
morphology, long-term stellar variability, observation gaps, and
cross-observatory consistency.

#### Period-proposal branch: approximately 1.5-2M parameters

A small auxiliary branch provides learned features analogous to
autocorrelation, Lomb-Scargle-like periodic features, box-least-squares-like
scores, and phase-folded summaries. These are proposal features, not final
scientific truth. They give the long-time model an explicit periodicity bias
without requiring it to rediscover every period-search method.

#### Spatial-temporal decoder: approximately 9-10M parameters

The decoder preserves FPN features and injects temporally enriched source
representations back into the spatial pyramid through deformable scatter. It
produces source, event, uncertainty, and wavelength-dependent maps.

To avoid emitting hundreds of full-resolution maps, wavelength outputs may use
a low-rank factorization:

```text
H[lambda, y, x] ≈ sum_k A[k, y, x] * B[k, lambda]
```

An initial rank of approximately 16-32 is a target for experimentation.

#### Global heads: approximately 2.5-3M parameters

Separate heads predict candidate/event classes, transit timing and shape,
orbital posteriors, uncertainty, artifact probability, data sufficiency, and
out-of-distribution status. The public v0 output is specified below; lensing
features remain an internal auxiliary representation.

### Parameter budget

| Component | Approximate parameters |
| --- | ---: |
| ConvNeXt-V2-style spatial encoder | 27M |
| FPN and source tokenizer | 3.5M |
| Wavelength/spectral encoder | 4M |
| Object graph/set encoder | 4.5-5M |
| Geometry and coverage encoder | 1M |
| Cross-modal fusion | 5-6M |
| Local Mamba-2 | 11-13M |
| Long-time Mamba-2 | 10-12M |
| Period proposal network | 1.5-2M |
| Spatial-temporal decoder | 9-10M |
| Prediction heads | 2.5-3M |
| **Total target** | **approximately 82-86M** |

The implementation must avoid flattening every spatial position, time step, and
wavelength into one sequence. The architecture factorizes the problem as
space, wavelength, cross-modal association, short time, and long time.

### 10.1 Dense wavelength-dependent heatmaps

For every long-time observation and every available short-time step, the
representation decoder should emit a spatial heatmap for each canonical
wavelength/energy bin:

```text
H[long_time, short_time, wavelength_bin, y, x, feature]
```

The dense feature channels should include at least:

```text
normalized signal or residual
transit-compatible signal score
source presence score
uncertainty
validity mask
interpolation mask
```

The internal wavelength encoder remains variable-length and wavelength-aware.
For dense storage and for the simple classifier, its output is decoded onto a
canonical logarithmic wavelength/energy grid. Each grid bin must carry an
availability and effective-response mask so an empty bin is not confused with
a measured zero signal.

Native variable-length wavelength tokens should remain available for training
and diagnostic use. Canonical heatmaps are a stable interchange representation,
not a claim that every observatory measured every bin at the same resolution.

The heatmap output should be written as chunked arrays, such as Zarr, and
referenced from the structured JSON result rather than embedded directly in
JSON.

### 10.2 Candidate extraction and simple event classification

The classifier pipeline should be:

```text
dense heatmaps
        ↓
spatial peaks/source proposals
        ↓
WCS-aware spatial association across observations
        ↓
per-source long/short/wavelength tracks
        ↓
simple event classifier
```

The first classifier should be a transparent baseline such as logistic
regression, a calibrated tree ensemble, or a small generalized additive model.
It should consume extracted features from the heatmaps rather than raw target
names or catalog identifiers.

Candidate-track features may include:

```text
signal depth and residual statistics by wavelength
short-time shape features
long-time recurrence features
number and spacing of possible events
uncertainty-weighted signal-to-noise
valid and interpolated sample counts
source isolation and neighboring-object context
coverage features
```

This separation gives us two independently inspectable artifacts:

```text
representation quality:
    do the heatmaps preserve source, wavelength, and time-dependent signals?

event-classifier quality:
    can a simple model distinguish transit-like events from artifacts?
```

## 10. Proposed model output

The output should be a structured set of candidate records rather than one
single classification label.

### 10.3 Candidate localization

For the uncentered patch:

```text
source_heatmap[y, x]
candidate_heatmap[y, x]
source confidence
candidate confidence
```

Each candidate should include both pixel coordinates and celestial coordinates
obtained through the patch WCS.

### 10.4 Event detection

For each candidate source:

```text
event_probability
event_midpoint posterior
event_start posterior
event_end posterior
event_duration posterior
transit/eclipse classification
event evidence score
```

The event score should be based on measured data and should be reported
separately from catalog priors and coverage quality.

Transit time is a required output, not an optional annotation. The result must
include an event-time posterior in a declared time system, preferably with
midpoint, start, end, and duration. A time-localization heatmap may also be
stored:

```text
event_time_heatmap[long_time, short_time, y, x]
```

The model should be able to report multiple candidate event windows for one
source when the observations are ambiguous.

### 10.5 Transit-shape parameters

When the short-time data support them:

```text
transit depth
transit duration
ingress duration
egress duration
planet-to-star radius ratio
impact parameter
```

These should be posterior distributions or calibrated intervals, not only
point estimates.

### 10.6 Orbital parameters

The orbital head may estimate:

```text
period posterior
reference transit epoch posterior
inclination posterior
scaled semimajor axis posterior
impact parameter posterior
eccentricity representation, when constrained
```

For eccentric orbits, `e*cos(omega)` and `e*sin(omega)` are preferable output
coordinates to raw eccentricity and argument of periastron when the data do not
separately constrain those parameters.

The model must emit an explicit constraint status for each orbital quantity:

```text
well_constrained
weakly_constrained
prior_dominated
unconstrained
```

An orbital-period estimate from one isolated transit must not be presented as
an observation-derived measurement.

### 10.7 Internal gravitational-lensing state

Lensing should remain an internal representation and auxiliary objective in the
first public output schema. It should not be folded into the ordinary transit
probability or required in every candidate record.

For diagnostics or later research releases, the internal branch may retain:

```text
microlensing probability
planetary microlensing perturbation probability
foreground lens candidate position
background source candidate position
event midpoint posterior
Einstein timescale posterior
impact parameter posterior
mass-ratio posterior
projected separation posterior
relative proper-motion posterior
astrometric separation or centroid-shift prediction
```

The internal lensing state should include proposed foreground/background roles
and their uncertainties. A planetary-lensing event may constrain a mass ratio
and projected separation without determining a complete orbital period or
three-dimensional orbit.

For strong or macrolensing diagnostics, the internal branch may instead include:

```text
strong-lensing probability
number of image/arcs hypotheses
lens-structure candidate position
source-structure candidate position
predicted image geometry
predicted time-delay representation, when measurable
```

The initial exoplanet search should prioritize planetary microlensing
perturbations as an auxiliary representation. Strong-lensing states are a
related future branch and should not be interpreted as exoplanet detections
without a planetary perturbation in the time-domain data.

### 10.8 Wavelength-dependent output

For multimodal spectral data, the output may include:

```text
transit depth by wavelength
spectral residual by wavelength
transmission-spectrum embedding
atmospheric-feature probabilities, in a later phase
```

The first version should focus on reconstructing wavelength-dependent transit
depth and uncertainty before attempting specific molecule classification.

### 10.9 Quality and uncertainty output

Every candidate record should include:

```text
aleatoric uncertainty
epistemic or ensemble uncertainty
coverage quality
missing-modality summary
out-of-distribution score
systematics/artifact probability
```

The model should be able to return:

```text
insufficient temporal coverage
insufficient wavelength coverage
signal dominated by systematics
out-of-distribution observation
```

### 10.10 Candidate record sketch

```json
{
  "source": {
    "x": 0.0,
    "y": 0.0,
    "ra_deg": 0.0,
    "dec_deg": 0.0
  },
  "detection": {
    "candidate_probability": 0.0,
    "event_evidence_score": 0.0,
    "artifact_probability": 0.0
  },
  "event": {
    "time_system": "BJD_TDB",
    "midpoint": {"median": null, "lower": null, "upper": null},
    "start": {"median": null, "lower": null, "upper": null},
    "end": {"median": null, "lower": null, "upper": null},
    "duration_seconds": {"median": null, "lower": null, "upper": null},
    "depth": {"median": 0.0, "lower": 0.0, "upper": 0.0}
  },
  "orbit": {
    "period": {"median": null, "lower": null, "upper": null},
    "inclination": {"median": null, "lower": null, "upper": null},
    "impact_parameter": {"median": null, "lower": null, "upper": null},
    "constraint_status": "unconstrained"
  },
  "spectral": {
    "wavelength_dependent_depth": [],
    "representation": []
  },
  "quality": {
    "coverage_score": 0.0,
    "missing_modalities": [],
    "out_of_distribution_score": 0.0
  }
}
```

## 11. Labels and training data

### 11.1 Synthetic pretraining and real-observation post-training

Synthetic observation bundles are the pretraining curriculum. Real Hubble
images and their associated calibration/engineering products are retained in
an external data store and reserved for post-training fine-tuning and held-out
evaluation. Large real-image payloads must not be committed to this
repository; manifests, provenance, hashes, and retrieval instructions are the
versioned interface.

The split boundary is by source/system and observation lineage. No exposure,
near-duplicate, or parent observation may cross from real calibration data into
the held-out evaluation set. Synthetic generation must expose a causal layer
manifest so astrophysical signal, instrument effects, orbital/pointing
behavior, timing corrections, and noise can be independently randomized or
replayed.

Synthetic pretraining must include realistic nuisance families rather than
only independent Gaussian noise. The HST-first curriculum should include
thermal focus breathing, pointing jitter/drift/roll, field-dependent PSF and
geometric aberration, UVIS CTE loss, IR persistence and nonlinearity, cosmic
rays, hot pixels, saturation, shutter effects, background/stray light, and
time-dependent sensitivity. Kepler/K2-derived pretraining domains should
include quarterly rolls, channel response, target-pixel/aperture changes,
thermal and pointing trends, impulsive systematics, and common-mode/cotrending
features. Each nuisance layer must carry an instrument/epoch and
measured-vs-approximation provenance label.

Timing and three-dimensional context must remain physically separated:
barycentric TDB/light-time and apparent-position corrections are timing and
geometry features; distance dilution, dust extinction, throughput, PSF losses,
detector losses, and saturation are photometric/instrument layers. A 3D
galaxy/scene map may condition distance, extinction, foreground/background
ordering, mass priors, and lens geometry, but must not become a catalog shortcut
for the planet label. Strong-field relativistic light bending or delay is an
optional, explicitly tiered effect and must not be fabricated for ordinary HST
fields where it is below the error budget.

Confirmed planets and host coordinates will be sourced from the NASA
Exoplanet Archive, especially its Planetary Systems and Planetary Systems
Composite Parameters tables.

Labels will combine:

```text
confirmed host/source location
known or predicted event windows
published orbital parameter priors
uncertainties and reference provenance
synthetic transit injections
synthetic stellar and planetary microlensing injections
foreground/background object-role labels where available
real non-host stars and artifact examples
```

Synthetic transits and microlensing events should be integrated over the actual
exposure duration and injected into real noise/systematics. The training set
should include shifted event windows, missing-wavelength cases, incomplete
object catalogs, and negative fields containing massive objects but no lensing
signal.

Splits must be made by host system or source, not by randomly splitting frames
from the same star across train and validation sets.

## 12. Initial implementation boundary

The first implementation should be data discovery and input validation, not
model training. It should:

1. Query a named star or sky patch through MAST REST services.
2. Build a manifest without downloading unnecessary products.
3. Expand selected product groups.
4. Inspect FITS headers and verify timestamps, exposure durations, WCS,
   wavelength information, uncertainty, and data-quality fields.
5. Build or retrieve a regional object list and validate object positions,
   separations, uncertainties, and available distance/proper-motion/mass
   information.
6. Produce a small normalized sample with long-time, short-time, wavelength,
   geometry, exposure, object-context, and coverage tensors.
7. Validate that missing modalities, interpolation masks, and incomplete object
   catalogs survive the whole preprocessing path.

The first model experiment should use a small target set with known HST
observations, synthetic injections, and a clearly documented train/validation
split.

## 13. Open research questions

- What fixed angular footprint gives useful source density without making the
  full patch prohibitively crowded?
- Which HST instruments and product types provide sufficiently dense temporal
  sequences for the first experiment?
- How should coverage quality be calibrated into detection uncertainty without
  becoming a label shortcut?
- Which spectral-response representation is available consistently enough for
  cross-observatory use?
- What minimum number and spacing of events is required before an orbital
  period is reported as data-constrained?
- Which object catalog or source-detection system provides sufficiently reliable
  positions, proper motions, distances, and mass priors for the lensing branch?
- How should foreground/background object roles be initialized when the catalog
  has no distance ordering?
- Which microlensing events have enough cadence and multi-observatory coverage
  for a meaningful planetary-perturbation training set?
- Should the first output be a source heatmap, coordinate-query records, or
  both in a shared model?

## 14. Implemented synthetic-v2 boundary

The repository now contains an opt-in parent-conditioned implementation in
addition to the bounded R0 generator. `RealObservationParent` preserves
loaded science, uncertainty, DQ, WCS/pointing, observer state, detector
history, and exact BJD_TDB exposure windows. `ObservationScheduleSampler`
replays those windows or resamples whole visits without inventing independent
cadence jitter.

`PopulationSampler` draws a coupled stellar/event system: stellar mass drives
bounded radius and temperature relations, orbital period and mass determine
semi-major axis, and transit duration is derived from the resulting geometry.
`PsfProvider` prefers caller-supplied empirical kernels and otherwise returns
an explicitly labeled wavelength/focus/position/jitter-aware optical
approximation. `WFC3UVISSimulator` and `WFC3IRSimulator` are separate; the IR
path includes nondestructive MULTIACCUM reads and history-dependent
persistence. `RealParentInjector` changes only the astrophysical source
signal and emits transit times, while preserving the parent’s uncertainty and
DQ arrays. `HubbleSyntheticV2` and
`iter_parented_synthetic_training_batches` expose this path to lazy model
training.

This implementation reaches the bounded R4 contract, not R5. RAW/IMA-level
injection through CRDS, `calwf3`, AstroDrizzle, external empirical PSF assets,
and real FITS loading require separately installed STScI software, reference
files, and archive data. Those external assets must be versioned by manifest,
hash, and provenance, not committed to Git. See
[`docs/HUBBLE_SYNTHETIC_V2.md`](docs/HUBBLE_SYNTHETIC_V2.md).
