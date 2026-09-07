# Fusion Engine — Final Technical Spec (Person 3)

*Filled in with actual measured/tuned values, per Hr 26-28 checklist task.
Matches the format of Gap 3's Technical Specification.*

## State Vector

```
x = [px, py, v, heading]
```
4-state EKF (position x/y in meters, speed, heading in radians).

*Note: the Gap Fixes spec's full design calls for a 6-state vector adding
`vx, vy` (world-frame velocity) and `gyro_bias`. This simplified 4-state
version was used for Phase 1 given the time constraints; the 6-state
upgrade is flagged as a future improvement (see Limitations below).*

## Process Noise (Q) — Actual Tuned Value

```python
Q = diag([0.5, 0.5, 0.3, 0.05]) * 150.0
  = diag([75.0, 75.0, 45.0, 7.5])
```

**Tuning methodology:** swept against the drift metric (not filter
innovation directly), per Gap 3's stated approach, in two stages:

1. **Synthetic stage:** tuned against a 20-second, 114-degree synthetic
   turn. Found the original placeholder Q (scale=1x) caused a 75% false
   GPS-rejection rate — heading estimation error compounding into large
   position drift during sharp turns, faster than Q let the filter's
   uncertainty grow to admit legitimate corrections.

2. **Real-data stage:** swept against 86 minutes / ~37km of real IO-VNBD
   S1 driving data (raw gyro-integration used as a stand-in for the AI
   model, since a trained checkpoint was not yet available). Measured
   error immediately BEFORE each GPS correction — i.e. genuine dead-
   reckoning drift accumulated over each real ~9-second gap between GPS
   fixes — rather than post-correction error, which is a misleading
   metric (a large enough Q makes the filter simply snap onto GPS at
   every fix, trivially producing near-zero post-correction error
   regardless of prediction quality in between).

**Result:** pre-correction error dropped from ~232m (scale=4x) to a floor
of ~73m starting at scale=150x, with GPS rejections eliminated entirely
(0%). Scaling Q beyond 150x gave no further improvement — that floor
reflects the real drift the raw-gyro-integration proxy accumulates in
~9 seconds, which Q cannot fix (Q only controls trust between prediction
and measurement, not the underlying prediction's quality).

## Measurement Noise — GPS (R_gps)

```python
R_gps = diag([3.0, 3.0])  # meters^2
```
Set from typical consumer GNSS horizontal accuracy (3-5m), per Gap 3's
guidance. Not re-tuned beyond the initial datasheet-based estimate in
Phase 1 — real GPS-vs-truth residuals were not separately available to
refine this further.

## Measurement Noise — Velocity/Heading (R_vel)

```python
R_vel = diag([0.5, 0.1])  # [ (m/s)^2, rad^2 ]  -- FALLBACK only
```
Used only when the AI model doesn't supply a per-prediction uncertainty.
When Person 2's trained TCN provides `velocity_log_variance` per the Gap 1
spec, R_vel's velocity term is overridden per-update:

```python
velocity_variance_ms2 = exp(velocity_log_variance) / 3.6**2
R = diag([velocity_variance_ms2, R_vel[1,1]])  # heading term stays fixed
```

*Not yet exercised against real model output, since a trained checkpoint
was not available during Phase 1 — verified only against the fallback
path using proxy (raw gyro-integration) inputs.*

## Failure-Mode Handling (Gap 5)

**Chi-squared innovation gating**, threshold = 95th percentile of the
chi-squared distribution for 2 degrees of freedom (~5.99). Rejects any
single update that's a statistical outlier given the filter's current
uncertainty.

**Rejection-streak safeguard** (identified as necessary during testing,
not present in the original spec): if 5 consecutive updates from the same
source (GPS or velocity) are rejected, the filter force-accepts the next
one regardless. Without this, testing showed the filter could enter
"lock" — a self-reinforcing loop where drift causes rejections, which
prevents the correction that would have fixed the drift.

**Measured outcome after tuning:** 0% GPS rejections and 0% velocity
rejections on both the synthetic test and the full real S1 dataset.

## Known Limitations / Honest Caveats for Final Submission

1. **Q is tuned against a proxy, not the real model.** The ~73m real-data
   drift floor reflects naive gyro-integration, not Person 2's trained
   TCN. Once a real checkpoint exists, Q must be re-swept — a better
   prediction model plausibly needs a *smaller* Q, since the filter would
   have less reason to distrust its own reasoning between GPS fixes.

2. **4-state, not 6-state.** No explicit `gyro_bias` term is tracked in the
   submitted system. An experimental 5-state variant with gyro_bias was
   built and tested (`ekf_fusion_with_gyro_bias.py`) against the same real
   S1 data. Result: only marginal improvement (73.3m -> 71.6m mean drift,
   ~2%), and the estimated bias converged to a physically implausible value
   (~0.56 rad/s, vs. real MEMS gyro bias which is typically ~0.001-0.01
   rad/s) -- indicating the term was absorbing other unmodeled errors
   (imperfect turn dynamics, sparse GPS) rather than isolating true sensor
   bias. This is a known, documented limitation in dead-reckoning
   literature: bias observability from position-only corrections spaced
   ~9 seconds apart is weak without an additional independent heading
   reference (e.g. magnetometer) or more frequent GPS. Given the marginal,
   unreliable gain, the simpler 4-state design was kept for submission.
   A follow-up worth exploring: constrain gyro_bias more tightly (smaller
   Q) or add a magnetometer-based heading update to properly separate
   bias from real motion.

3. **R_gps was not empirically re-tuned** against real GPS residuals in
   Phase 1 -- it uses the datasheet-typical value only.

4. **GraphHopper map-matching integration is code-complete but untested
   against a live server** — no real UK OpenStreetMap-backed GraphHopper
   instance was stood up during Phase 1 due to time constraints. Verified
   instead against a simpler nearest-point stand-in matcher, which proved
   the data pipeline (EKF output -> coordinate conversion -> matcher)
   works correctly, but does not exercise the actual Hidden-Markov-based
   matching logic the final design calls for.
