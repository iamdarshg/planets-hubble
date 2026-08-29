# Planets-Hubble: Multimodal Exoplanet Discovery Specification

Status: design draft

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

## 9. Proposed model architecture

```text
1280x720 patch encoder
        ↓
spatial feature map

wavelength-token encoder
        ↓
hyperspectral representation

exposure and observer-geometry encoder
        ↓
observation-state representation

short-time event encoder
        ↓
local event representation

long-time encoder
        ↓
long-baseline recurrence representation

coverage and quality branch
        ↓
observation-quality representation

cross-modal fusion
        ↓
prediction heads
```

The full 1280x720 patch remains the input. Internally, a multiscale spatial
encoder may first detect sources at coarse resolution and then extract
high-resolution features around candidate sources.

## 10. Proposed model output

The output should be a structured set of candidate records rather than one
single classification label.

### 10.1 Candidate localization

For the uncentered patch:

```text
source_heatmap[y, x]
candidate_heatmap[y, x]
source confidence
candidate confidence
```

Each candidate should include both pixel coordinates and celestial coordinates
obtained through the patch WCS.

### 10.2 Event detection

For each candidate source:

```text
event_probability
event_midpoint posterior
event_start posterior
event_end posterior
transit/eclipse classification
event evidence score
```

The event score should be based on measured data and should be reported
separately from catalog priors and coverage quality.

### 10.3 Transit-shape parameters

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

### 10.4 Orbital parameters

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

### 10.5 Wavelength-dependent output

For multimodal spectral data, the output may include:

```text
transit depth by wavelength
spectral residual by wavelength
transmission-spectrum embedding
atmospheric-feature probabilities, in a later phase
```

The first version should focus on reconstructing wavelength-dependent transit
depth and uncertainty before attempting specific molecule classification.

### 10.6 Quality and uncertainty output

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

### 10.7 Candidate record sketch

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
    "midpoint": {"median": 0.0, "lower": 0.0, "upper": 0.0},
    "duration": {"median": 0.0, "lower": 0.0, "upper": 0.0},
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
real non-host stars and artifact examples
```

Synthetic transits should be integrated over the actual exposure duration and
injected into real noise/systematics. The training set should include shifted
event windows and missing-wavelength cases.

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
5. Produce a small normalized sample with long-time, short-time, wavelength,
   geometry, exposure, and coverage tensors.
6. Validate that missing modalities and interpolation masks survive the whole
   preprocessing path.

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
- Should the first output be a source heatmap, coordinate-query records, or
  both in a shared model?
