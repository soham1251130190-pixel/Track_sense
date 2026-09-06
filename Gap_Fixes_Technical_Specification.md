# Technical Specification — Closing Gaps 1, 3, 4, 5, 6

This document gives concrete, defensible specifications for every gap that doesn't depend on running against real data. Gap 2 (preliminary results) is addressed separately in `tcn_ekf_evaluation.py` — a complete, tested, ready-to-run script — because that gap genuinely requires executing code against the real dataset, and fabricating numbers here would be worse than the gap itself.

---

## Gap 1 — Model Architecture

**Architecture: a causal, dilated Temporal Convolutional Network (TCN).**

Causal convolutions are the right choice specifically *because* the system runs in real time — a causal layer only ever looks at past timesteps, never future ones, which matches how the model will actually be fed data live (it can't peek ahead). This isn't just "a TCN" chosen generically; it's chosen because the causality constraint is architecturally identical to the real-time inference constraint.

**Exact specification:**

| Parameter | Value |
|---|---|
| Layers | 5 stacked TCN residual blocks |
| Dilation schedule | 1, 2, 4, 8, 16 (doubling per layer — standard TCN receptive-field growth) |
| Channels per layer | 64 |
| Kernel size | 3 |
| Residual connections | Yes, per block (1×1 conv projection when channel count changes) |
| Dropout | 0.1, applied after each conv in a block |
| Pooling | Global average pooling over the time axis after the last TCN block |
| Head | Dense(64→32) → ReLU → Dense(32→3) |

**Input:**
- 6-axis IMU: accelerometer (x, y, z) + gyroscope (x, y, z)
- Window size: 100 timesteps at a 10Hz sample rate (10 seconds of context) — chosen to match the sample rate of the public benchmark dataset being used for pretraining
- Preprocessing: per-axis z-score normalization using running statistics; gravity-compensated accelerometer (subtract the gravity component estimated from a short static calibration period or from a complementary filter using the magnetometer)

**Output — not a single point estimate:**
The model outputs three values: `(velocity_mean, velocity_log_variance, heading_rate)`. The log-variance term is the important design choice: it means the model reports its own uncertainty alongside its prediction, which is what allows the Kalman filter (Gap 3) to trust confident predictions more than uncertain ones, rather than treating every prediction as equally reliable.

**Loss function:** Gaussian negative log-likelihood on `(velocity_mean, velocity_log_variance)` against ground-truth speed (available from GPS during non-blackout portions of training data), plus an auxiliary L1 loss on heading rate.

**Training strategy:**
- Optimizer: AdamW, initial learning rate 1e-3, cosine annealing schedule
- Batch size: 256
- ~50 epochs, with early stopping tracked on validation-set **drift metric** (not just validation loss) — this matters because a model can have a good loss number while still drifting badly in practice; the metric that matters is the one in Gap 2's evaluation.

**Model size (calculated from the architecture above, not yet benchmarked on-device):**
- Approximately 104,000 parameters
- ~406 KB at float32
- ~101 KB after INT8 quantization

**Inference latency:** Not yet benchmarked on target hardware — stating a specific millisecond figure without having run it would be exactly the kind of unsupported claim this feedback is pushing back on. What can be said honestly: a model this size (≈100K parameters, 5 causal conv layers) is well within the range that runs in single-digit milliseconds on a mid-range mobile CPU via TensorFlow Lite with the XNNPACK delegate, based on published benchmarks of similarly-sized TCNs — but the actual number must be measured on the real target device before it's quoted in a final submission.

---

## Gap 3 — Kalman Filter Integration (Full Detail)

**State vector** (6-dimensional, kept deliberately smaller than a full 15-16 state INS filter since there is no wheel-speed input to justify tracking full 3D attitude and all bias terms with high confidence):

```
x = [x, y, vx, vy, heading, gyro_bias]
```

- `x, y` — position in a local tangent-plane frame (meters)
- `vx, vy` — world-frame velocity components
- `heading` — vehicle heading angle
- `gyro_bias` — slowly-varying gyroscope bias, estimated online

**Process model** (runs every IMU sample, ~10Hz+):
```
heading[k]  = heading[k-1] + (gyro_z - gyro_bias) * dt
vx[k], vy[k] = rotate(accel_x, accel_y, heading[k]) integrated over dt
x[k], y[k]   = previous position + velocity * dt
```
This is a nonlinear kinematic model (not simple constant-velocity) — acceleration is measured directly from the IMU and rotated into the world frame using the current heading estimate, rather than assumed constant. The process noise covariance `Q` is a diagonal matrix, with larger uncertainty injected into `vx, vy` (since accelerometer integration is the biggest error source) and very small uncertainty on `gyro_bias` (which should drift slowly).

**Measurement models — there are two, used in different situations:**

1. **GNSS position update** (when GNSS is available): `H` maps state directly to `(x, y)`; measurement noise `R` set from the GNSS receiver's reported horizontal accuracy (typically 3–5m for consumer GNSS).

2. **AI-predicted velocity update** (used during GNSS blackout — this is the actual mechanism that makes the TCN useful): `H` maps state to `(vx, vy)`. The TCN's scalar speed prediction is projected into world-frame `(vx, vy)` using the filter's *current* heading estimate. The measurement noise `R` is **not fixed** — it's set directly from the TCN's own predicted log-variance for that window. This is the concrete link between Gap 1 and Gap 3: an uncertain TCN prediction automatically gets down-weighted by the filter, rather than corrupting the state estimate as confidently as a certain one would.

**Update equations** are the standard EKF innovation/gain/correction form:
```
y = z - H·x                    (innovation)
S = H·P·H^T + R                (innovation covariance)
K = P·H^T·S^-1                 (Kalman gain)
x = x + K·y                    (state correction)
P = (I - K·H)·P                (covariance correction)
```

**Covariance tuning approach:** `Q` (process noise) is initially set from the sensor datasheet's noise specification, then tuned empirically against the validation set from Gap 2's evaluation loop — the tuning target is minimizing the drift metric, not minimizing filter innovation directly, since those two objectives can diverge.

A working, tested implementation of this exact filter (both the GNSS and velocity update paths) is in `tcn_ekf_evaluation.py`.

---

## Gap 4 — Edge-Deployable Engine

**Target platform:** NVIDIA Jetson Nano (or Jetson Orin Nano if higher throughput is needed) — chosen because it's a realistic, obtainable dev-kit-class edge device with a mature TensorRT deployment path, rather than a purely hypothetical "custom hardware" claim.

**Interface for external IMU input:** A lightweight message-passing interface using **ZeroMQ with Protobuf-serialized messages**, so any IMU with a driver that can publish to this interface can be connected — this avoids hard-coding a single sensor vendor. For a more standard robotics-adjacent option, ROS2 topics are a drop-in alternative, since Jetson devices commonly run ROS2 already.

**What's different from the mobile version:**
- The edge engine can use a **higher-grade IMU** (e.g., a FOG-based or tactical-grade MEMS unit) rather than a phone's consumer sensor, which is *why* it can sustain a higher update rate (~200Hz) — the phone version is capped around 10Hz partly by Android's sensor batching behavior and partly by matching the training data's native sample rate.
- Model inference uses **TensorRT** acceleration (rather than TensorFlow Lite on mobile), since TensorRT is the standard optimized runtime for NVIDIA edge hardware and typically achieves lower latency than a generic mobile runtime on the same silicon.
- The edge engine has more compute headroom, so it could run a slightly larger model variant or an ensemble of the TCN and a secondary model, if evaluation shows that improves accuracy — this is a possible future extension, not yet implemented.

**Performance claim, stated honestly:** a target of low-single-digit-millisecond inference is realistic for a ~100K-parameter model on Jetson-class hardware with TensorRT, but — consistent with Gap 1's inference latency section — this needs to be actually measured on the target board before being stated as a fixed number in a final submission.

---

## Gap 5 — Failure Mode Analysis

**What happens when the AI makes a wrong prediction?**
The uncertainty-aware fusion design (Gap 3) is the primary safeguard: a bad prediction typically comes with a high predicted variance (the model has less "confidence" in unfamiliar conditions), which automatically reduces its influence on the filter via the `R` term. As a second layer of defense, an **innovation gating check** (a chi-squared test on the innovation `y` relative to its covariance `S`) should reject any single update that's a statistical outlier — e.g., a spurious velocity prediction wildly inconsistent with the current state — rather than blindly applying it.

**How does the system recover from poor initialization?**
Dead-reckoning mode should not begin from an uncertain state. The design requires a **minimum-confidence GNSS fix** (a maximum horizontal accuracy threshold) before allowing a transition into dead-reckoning mode, so the filter always starts a blackout period from a well-known position and velocity. For heading initialization specifically, a short coarse-alignment step using the accelerometer's gravity vector plus magnetometer heading is used before the vehicle starts moving, rather than relying on the gyroscope's integrated heading from an arbitrary starting point.

**What's the worst-case drift?**
Rather than letting the filter report a position with false confidence indefinitely, the system should track its own **growing uncertainty** (the covariance `P`, specifically the position sub-block) throughout a blackout. If that uncertainty crosses a defined threshold, the UI should visibly degrade — e.g., showing an expanding "uncertainty radius" around the position indicator, or falling back to a simpler "last known road segment" indicator — rather than continuing to show a precise-looking dot that may be significantly wrong. This is a UX decision as much as an algorithmic one, but it directly follows from having an uncertainty estimate available at all.

**How does performance degrade with phone orientation changes mid-drive?**
The calibration engine (component 1 in the system architecture) is not a one-time startup step — it should continuously monitor whether the accelerometer's gravity-vector direction has shifted significantly (e.g., the phone was picked up, repositioned, or fell out of its mount), and automatically re-trigger a recalibration if so, rather than silently continuing to apply a now-stale orientation transform.

---

## Gap 6 — Data Augmentation Strategy

The core issue this addresses: the public benchmark dataset (IO-VNBD) was recorded on UK, Nigerian, and French roads, and does not necessarily represent target-region-specific conditions (e.g., dense mixed traffic, two-wheelers, unmarked lanes, frequent potholes). Rather than assuming the model will generalize for free, the training pipeline should apply:

- **Sensor gain/scale augmentation** — randomly scale each accelerometer/gyroscope axis by a small factor (e.g., ±5%) per training sample, to simulate the fact that different phone models' IMU chips have slightly different sensitivity characteristics, so the model doesn't overfit to one specific test device's calibration.

- **Noise injection** — add both continuous Gaussian noise (simulating baseline sensor noise floor) and occasional sharp impulsive noise bursts (simulating potholes and road joints), since these are qualitatively different disturbances that a model trained only on smooth Gaussian noise may not generalize to.

- **Simulated vibration overlay** — superimpose real high-frequency vibration signatures (recorded once from an idling/moving vehicle) at randomized amplitude onto training windows, rather than relying only on whatever vibration happens to already be present in the base dataset's recordings.

- **Simulated mounting-orientation variation** — apply a random static rotation offset to the accelerometer/gyroscope axes for each training sample, simulating different phone mounting angles (dashboard mount vs. windshield mount vs. cupholder). This is what teaches the calibration engine (and the TCN) to be robust to residual misalignment rather than assuming a single fixed mounting geometry.

- **A small amount of locally-recorded calibration data** — rather than assuming the public dataset alone is sufficient, even a modest amount of self-recorded local driving data (on the order of an hour) should be used for fine-tuning or updating the model's input normalization statistics, given realistic time constraints. This is a pragmatic middle ground between "train only on the public dataset" (risks poor local generalization) and "collect a full local dataset from scratch" (not realistic in the available time).
