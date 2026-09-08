# TrackSense — AI-Based Navigation for Signal-Denied Environments

## 1. One-Line Summary

TrackSense is a mobile app plus an on-device AI engine that keeps a vehicle's live position accurate even when satellite positioning (GPS/GNSS) signal is completely unavailable — inside tunnels, underground parking structures, dense urban canyons between tall buildings, and other blackout zones — by fusing the phone's own motion sensors through a trained AI model instead of relying on external hardware or a network connection.

---

## 2. The Problem, In Detail

Smartphone navigation apps rely entirely on GNSS (GPS, Galileo, NavIC, etc.) for positioning. In several common real-world scenarios, that signal disappears completely:

- Long road/rail tunnels
- Multi-level underground parking structures
- Dense forested highways
- Deep urban canyons between skyscrapers

When this happens, current navigation apps **freeze, jump erratically, or miscalculate upcoming turns**. This causes missed exits, delivery delays, and safety hazards — particularly for logistics fleets, ride-hailing drivers, and quick-commerce delivery vehicles operating in dense cities.

Most vehicles on the road — commercial trucks, older cars, and the large majority of two-wheelers — have **no factory-fitted Inertial Navigation System (INS)** wired into the vehicle. They rely solely on a smartphone mounted on the dashboard or in a holder. Using that phone's own sensors (accelerometer + gyroscope, together called an IMU — Inertial Measurement Unit) to estimate position without GNSS is called **dead reckoning**, and it is difficult for several concrete reasons:

- The phone is subjected to chassis vibration, engine harmonics, sudden braking, and road potholes — all of which corrupt the raw sensor signal
- There is no wheel-speed (OBD-II) feed to cross-check estimated speed against, unlike a factory INS
- Small sensor bias and noise errors **compound exponentially** through the mathematical integration needed to go from acceleration → velocity → position, so a naive dead-reckoning estimate drifts away from the true position within seconds of losing GNSS

**Important distinction that shaped the whole design:** GPS itself does *not* require an internet connection — a GPS chip is a radio receiver that listens directly to satellite signals. What internet is normally used for is (a) downloading satellite orbit data to get a faster initial fix (A-GPS), and (b) downloading map tiles/images to draw the visual map. The actual failure mode this project solves is **loss of the satellite signal itself** (blocked by a physical structure), not loss of internet connectivity. This is a physics problem, not a connectivity problem, and the two are commonly confused.

**Also distinct from existing tools:** Google Maps' "offline maps" feature only solves the *internet connectivity* problem (pre-downloaded map tiles, GPS still used normally for positioning). It does **not** solve the signal-blackout problem — offline mode behaves exactly like online mode when GPS itself is lost, aside from a very crude, non-AI heuristic that interpolates a known route for a few seconds. That heuristic is not AI/ML-based, does not filter sensor noise with a trained model, does not do GNSS+INS fusion, and does not hit any quantified accuracy target — it degrades quickly in anything longer than a very short gap. This project's approach is meaningfully different and more robust.

---

## 3. The Solution — What Is Being Built

A lightweight, edge-deployable software engine and companion mobile app that turns a smartphone into a self-contained **Intelligent Dead Reckoning (IDR) system with GNSS fusion**. It has six required components:

1. **In-Vehicle Alignment & Calibration Engine** — automatically detects the phone's pitch/roll/yaw relative to the vehicle's actual direction of travel, regardless of how or where the phone is mounted (dashboard, holder, pocket), without requiring the driver to manually position it.

2. **AI Speed & Vibration Filter** — a trained machine learning model that filters out non-navigation motion (idling vibration, potholes, the phone shifting in its mount) and estimates the vehicle's true velocity and acceleration from noisy IMU data alone. This is the core innovation: instead of a fixed mathematical formula, the model *learns* real vibration/noise patterns from actual recorded driving data.

3. **Map-Matching with Kinematic Constraints** — snaps the drifting inertial position estimate onto the real road network (using OpenStreetMap data) while respecting physical rules a vehicle must obey — e.g., it cannot slide sideways or fly upward, and it must follow road direction and turn geometry (these are called non-holonomic constraints).

4. **GNSS+INS Fusion Engine** — an AI-assisted sensor fusion algorithm (built around an Extended Kalman Filter, a well-established estimation technique) that combines the GNSS signal (when available) with the AI-corrected inertial estimate to produce the most accurate possible position and eliminate drift.

5. **Seamless Mode-Switching** — an instant, smooth transition between "GNSS-aided" mode and "AI dead-reckoning" mode, and back again, that happens within milliseconds so the user never sees the position freeze, jump, or visibly re-snap.

6. **Real-Time Navigation Interface** — the actual mobile app UI, showing a smooth, uninterrupted vehicle position indicator with continuous navigation guidance throughout.

### How it addresses the problem
- Removes the frozen/jumping navigation screen inside tunnels and underground parking
- Reduces missed exits and wrong turns for delivery, ride-hailing, and logistics fleets operating in dense cities
- Provides a continuous, trustworthy position estimate instead of the app guessing, or snapping incorrectly once signal returns

### What makes it different from existing approaches
- Most existing navigation apps fall back on simple, fixed-formula dead reckoning that drifts quickly and is not tuned to how a real vehicle actually moves. This project's engine instead **learns** vibration and noise patterns from real driving data, adapting to actual vehicle behavior.
- Runs **entirely on the phone** — no external hardware (no OBD-II wire), and critically, **no cloud dependency at runtime** — which matters because GNSS-denied zones very often also have poor or no internet connectivity, so any cloud-dependent design would fail exactly where it's needed most.
- Includes an automatic calibration step that determines phone orientation/mount position without requiring manual setup from the driver.

---

## 4. System Architecture (Data Flow)

```
Smartphone Sensors
 ├─ Accelerometer   ─┐
 ├─ Gyroscope        ├─→ In-Vehicle Alignment & Calibration Engine
 ├─ Magnetometer    ─┘              │
 │                                   ▼
 │                     AI Speed & Vibration Filter (ML Model)
 │                                   │
 └─ GNSS Receiver ──────────────────▶│
       (GPS/Galileo/NavIC)           ▼
                        GNSS + INS AI Fusion Engine
                                     │
                                     ▼
                Map-Matching Engine (OpenStreetMap + Kinematic Constraints)
                                     │
                                     ▼
                     Real-Time Position & Navigation UI
```

## 5. Operational Flow — Mode Switching Logic

```
Navigation Active
       │
       ▼
GNSS Signal Available? ──── yes ──→ GNSS-Aided INS Mode (direct satellite fix + IMU refinement)
       │
       no (tunnel / underground / canyon)
       │
       ▼
Dead Reckoning Mode (IMU-only tracking)
       │
       ▼
AI filters noise (vibration, potholes, engine idle)
       │
       ▼
Map-match trajectory onto road network
       │
       ▼
Continuous Position Output to UI
       │
       ▼
Re-check GNSS every cycle (loop, <100ms) ──→ back to "GNSS Signal Available?"
```

## 6. Training & Deployment Workflow

The workflow is split into two phases:

**Phase 1 — Cloud/Desktop (before deployment)**
1. Collect a dataset combining a public benchmark dataset (see Section 8) with self-recorded local driving data
2. Train the AI models (speed estimation, sensor fusion correction, map-matching support) using this data
3. Optimize and export the trained model — quantization, pruning, and conversion to a mobile-friendly format (ONNX or TensorFlow Lite) to shrink it for real-time use

**Phase 2 — On-Device (real-time operation)**
1. Load the lightweight, optimized model onto the smartphone
2. Feed it the live IMU + GNSS sensor stream in real time
3. Run real-time inference — target 10Hz update rate on a standard smartphone, or roughly 200Hz on a dedicated edge-deployable engine using higher-grade IMU sensors

This split matters because the heavy computation (training) happens beforehand, so that what actually runs live on the phone is fast and lightweight — no cloud round-trip needed during operation.

---

## 7. Performance Targets

| Metric | Target |
|---|---|
| Dead-reckoning positional drift | Less than 10% of the distance traveled during a GNSS blackout — e.g., under 5 meters of drift over 50 meters traveled in under a minute, or under 100 meters of drift over a 1km tunnel at 60 km/h |
| Position update rate | 10Hz on a standard smartphone app; approximately 200Hz on a dedicated edge-deployable engine with higher-grade (e.g. fiber-optic gyroscope based) IMU sensors |

For context: this drift target is comparable to what automotive-grade INS systems (which have the advantage of a wired wheel-speed sensor feed) struggle to consistently achieve. Hitting it using only a smartphone sitting in a cupholder, with no external hardware, is a genuinely difficult, research-adjacent target rather than a routine engineering task.

---

## 8. Public Dataset Used for Training

**IO-VNBD (Inertial and Odometry Vehicle Navigation Benchmark Dataset)**
- Originally published by researchers at Coventry University (2021)
- Contains data from a research vehicle's GPS receiver, inertial sensors, and wheel-speed sensors, **plus** inertial sensors and GPS from an Android smartphone sampled at 10Hz
- Scale: roughly 40 hours / 1,300km of vehicle-sensor data, and roughly 58 hours / 4,400km of smartphone-sensor data (a later expanded version reportedly reaches around 98 hours / 5,700km)
- Recorded across a range of real-world conditions: traffic congestion, roundabouts, hard braking, wet/gravel/country roads, and sloped roads
- **Limitation to note:** this dataset was recorded on UK, Nigerian, and French roads — not necessarily representative of every target region's road conditions (potholes, mixed traffic patterns, unmarked lanes, auto-rickshaws/scooters, etc.). Supplementing it with a modest amount of self-recorded local driving data was identified as an important step to adapt the model to local conditions.

---

## 9. Technology Stack (What's Used, and Where)

| Component | Technology | Purpose |
|---|---|---|
| Mobile application | Android (Kotlin) | The actual navigation app the driver uses |
| Raw sensor access | Android `SensorManager` API | Captures real-time accelerometer, gyroscope, and magnetometer data from the phone |
| Model training | Python + PyTorch | Where the AI model is trained offline, using the dataset described above |
| On-device inference | TensorFlow Lite / ONNX Runtime Mobile | Runs the trained model efficiently on the phone itself, after quantization/pruning to shrink its size and speed |
| Sensor fusion / filtering | Extended Kalman Filter (prototyped with the Python `filterpy` library, then reimplemented natively for the app) | Mathematically combines the AI-corrected inertial estimate with any available GNSS reading into one best position estimate |
| Map data | OpenStreetMap (OSM) | Free, open, downloadable road network data — used instead of a proprietary mapping service (see Section 11 for why) |
| Map-matching engine | GraphHopper or Valhalla (specifically their map-matching modules, sometimes referred to by the underlying technique, Hidden Markov Model map-matching) | Snaps the estimated (and potentially drifting) trajectory onto the actual road network, correctly resolving ambiguous cases like parallel roads or flyovers |

---

## 10. Why This Uses OpenStreetMap Instead of a Commercial Maps API

This was a deliberate architectural decision, not a default choice, directly tied to the core constraint of the problem:

| | OpenStreetMap-based approach (used here) | A typical commercial maps platform |
|---|---|---|
| Cost | Free, no API key or billing account needed | Usually requires an account, API key, and has a paid usage tier |
| Map-matching | Can run locally / offline, self-hostable | Snap-to-road type features are typically a paid, internet-dependent cloud API call |
| Fits an "on-device, no cloud dependency" design | Yes — road data can be downloaded and matched entirely locally | No — this would reintroduce exactly the connectivity dependency the project is trying to eliminate |

The core reasoning: the target scenario (GNSS-denied zones) very often *also* has degraded internet connectivity. If the map-matching step depended on a live cloud API call, the system would fail in precisely the situation it was built to handle. This is treated as a key point of technical credibility for the project, since it shows the architecture was designed around its actual constraint rather than just wired to whatever mapping tool was convenient.

Note: for a **visual demo/presentation prototype only** (not the real production system), using a free map-tile rendering service for the visuals is acceptable, since that only affects how the map is drawn on screen, not the actual positioning/map-matching logic being demonstrated.

---

## 11. Prior / Related Work Reviewed

This is an active area of research, not an unsolved-in-theory problem — the real difficulty is achieving the accuracy target on ordinary phone hardware in real time, not inventing the concept from scratch. Key references reviewed:

1. **A 2025 peer-reviewed paper** proposing a neural network (referred to as AVNet in the paper) that estimates vehicle attitude and velocity from smartphone sensors, combined with an invariant Extended Kalman Filter, for smartphone-only dead reckoning in GNSS-denied areas. Demonstrated in real conditions like tunnels and underground parking lots. This is the closest published blueprint to the architecture being used here — a neural network for motion-cue extraction feeding into a Kalman-filter-based fusion stage.

2. **"AI-IMU Dead-Reckoning" (2020)** — combines a Physics-Informed Neural Network (a network that has physical motion constraints built into its structure) with an Extended Kalman Filter. Reportedly achieves accuracy close to LiDAR-based localization methods using only IMU data — a strong proof point that AI-corrected dead reckoning can approach the performance of much more expensive sensor setups.

3. **Lane-detection-aided dead reckoning (KAIST, 2021)** — fuses inertial navigation with a learning-based lane-detection model (using the vehicle's camera) via an Unscented Kalman Filter, to further bound position drift. Relevant as a possible future extension if camera input is added as a secondary correction signal.

4. **Industry baseline (a semiconductor manufacturer's public technical writeup)** — describes GNSS+dead-reckoning sensor fusion, typically via Kalman/Extended Kalman filters, as already-standard practice in commercial GPS chipsets for handling tunnels and urban canyons. This represents the baseline that this project's AI-enhanced approach is intended to outperform, not something being invented in isolation.

**What existing work provides**, versus **what remains genuinely open**:

| Existing work provides | What's still open / the actual contribution here |
|---|---|
| A proven architecture pattern: neural network for motion-cue extraction + Kalman-filter-based fusion | Making that architecture run in real time on an ordinary phone CPU (most published research benchmarks on desktop/GPU hardware) |
| A benchmarking methodology and public dataset | Hitting the specific drift target under **locally relevant** road conditions (which the benchmark dataset does not fully represent) |
| An established industry baseline (classical Kalman-filter fusion) | Building the full pipeline end-to-end — the AI model *and* the map-matching layer *and* a working mobile app, not just an isolated ML model |

---

## 12. Why This Is a Genuinely Hard Engineering Problem

- **Compounding error** — small gyroscope/accelerometer bias errors multiply through the mathematical integration steps (rate → orientation → velocity → position), causing rapid drift if uncorrected
- **No ground truth to correct against in real time** — there is no wheel-speed feed; the AI has to distinguish genuine vehicle motion from vibration/noise using only the noisy sensor stream itself, with nothing to check its answer against as it runs
- **Hardware variability** — every phone model's IMU chip has different noise and bias characteristics, and the same system has to work across cars, trucks, and two-wheelers, each with very different vibration and motion profiles
- **Map-matching ambiguity** — intersections, closely parallel roads, and flyovers stacked over surface roads make "snap to the nearest road" a genuinely hard probabilistic reasoning problem, not a simple geometric lookup
- **Real-time, fully on-device constraint** — the entire pipeline (noise filtering, velocity estimation, sensor fusion, map-matching) has to run locally on a phone at 10Hz or faster, without draining the battery or lagging, and specifically without cloud offload (since connectivity is often also poor exactly where this matters)
- **A strict, quantitative accuracy bar** — the target drift ceiling is comparable to what professional automotive-grade INS systems (which have a wired wheel-speed input) struggle to consistently achieve; doing it from a phone with no external hardware is a genuinely open, research-adjacent challenge rather than routine engineering

---

## 13. Suggested Team Role Split (5–6 people)

| Role | Focus |
|---|---|
| ML Model (1–2 people) | Train and tune the neural network + Kalman-filter fusion pipeline in Python |
| Mobile App (1 person) | Android UI, sensor capture via `SensorManager`, integrating the on-device ML model |
| Map-Matching Integration (1 person) | Wiring GraphHopper/Valhalla into the app, handling the OpenStreetMap data |
| Data Collection & Testing (1 person) | Recording driving data, running evaluation against the drift-accuracy target |
| Documentation & Presentation (1 person) | Solution writeups, diagrams, and presentation materials |

---

## 14. Demonstration Prototype (For Live Presentation)

Because the real system requires physical driving data and a real mobile device to demonstrate properly, a lightweight **visual simulation** was designed specifically for presenting the concept live:

- A web-based map view (using an open map-tile renderer, e.g. Leaflet.js with OpenStreetMap tiles) showing a vehicle icon moving along a real road route
- A toggle button simulating "entering a tunnel" — i.e., simulating GNSS signal loss
- On triggering signal loss, the demo shows **two vehicle position indicators simultaneously**, to visually contrast the difference the project makes:
  - A "naive dead reckoning" indicator that visibly drifts off the road (simulated with an uncorrected random-walk noise model)
  - A "TrackSense AI + map-matching" indicator that stays snapped correctly to the road (simulated with a corrective/smoothing step tied to the known route)
- A live stats panel showing current mode (GNSS-aided vs. AI dead-reckoning), a real-time drift distance measurement between the two indicators, and a small chart plotting drift-over-time during the simulated blackout
- On "exiting the tunnel," both indicators reconcile with the real GPS position — deliberately showing the naive version snapping back abruptly versus the AI version transitioning smoothly

This is explicitly a **presentation aid** — a simplified visual simulation to make the core idea intuitive to an audience quickly — and is not the real sensor-fusion implementation itself.

---

## 15. Status Summary — What Has Been Done So Far

- Problem framing and technical scope fully defined (Sections 2–3 above)
- Full system architecture diagrammed: end-to-end data flow, mode-switching decision flow, and the two-phase training/deployment workflow (Sections 4–6)
- Performance targets and success metrics defined (Section 7)
- Public benchmark dataset identified and its limitations understood (Section 8)
- Complete technology stack selected, with the OpenStreetMap-vs-commercial-API decision explicitly reasoned through rather than defaulted (Sections 9–10)
- Prior academic and industry work researched and reviewed, situating this project relative to the state of the art rather than reinventing from zero (Section 11)
- Key technical risks and difficulty drivers identified upfront (Section 12)
- Suggested team structure proposed for parallel execution (Section 13)
- A presentation-ready visual demo concept designed, with a full build prompt ready to hand to a coding agent (Section 14)
- Presentation materials in progress: a proposed-solution/technical-approach visual layout, and supporting explanatory materials for a live audience

**Not yet done:** the actual production mobile app, the trained AI model itself, and self-recorded local driving data collection — these remain as build work ahead.
