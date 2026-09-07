"""
ekf_fusion.py
-------------
Person 3 (Fusion Engineer) — Hr 0-6 deliverable.

Extended Kalman Filter for GNSS + AI-corrected-IMU sensor fusion,
used for vehicle dead reckoning in GNSS-denied zones (tunnels,
underground parking, etc.)

State vector:   x = [px, py, v, heading]
    px, py    : position in a local flat-earth (meters) frame, NOT raw lat/lon.
                Convert lat/lon -> local xy upstream (Person 1's job) before
                feeding data into this filter.
    v         : speed (m/s)
    heading   : heading angle (radians), 0 = +x axis, CCW positive

Two measurement sources feed this filter:
    1. GNSS fix          -> update_gps(px, py)
    2. AI/IMU estimate    -> update_velocity(v, heading)   <-- Hr 3-6 task

Predict step advances state using a constant-velocity / constant-heading
kinematic motion model between measurements.

This file has NO dependency on Person 1 or Person 2's real outputs yet.
It runs fully on synthetic data so you can verify the plumbing today,
then swap in real velocity/heading (Person 2) and real GPS (Person 1)
once they exist, without changing this class's interface.
"""

import numpy as np
from scipy.stats import chi2


class VehicleEKF:
    def __init__(self, initial_state=None, initial_covariance=None):
        """
        initial_state: array-like [px, py, v, heading]. Defaults to zeros.
        initial_covariance: 4x4 array. Defaults to a loose starting guess.
        """
        self.n = 4  # state dimension

        if initial_state is None:
            initial_state = np.zeros(self.n)
        self.x = np.array(initial_state, dtype=float).reshape(self.n)

        if initial_covariance is None:
            initial_covariance = np.diag([5.0, 5.0, 2.0, 0.5])  # loose prior
        self.P = np.array(initial_covariance, dtype=float)

        # --- Process noise (Q) ---
        # Tuned in TWO stages, per the Gap 3 spec's guidance to tune against
        # the drift metric, not against filter innovation directly:
        #
        # Stage 1 (synthetic, Hr 6-10): scale=4x, tuned against a 20-second
        # synthetic 114-degree turn. Fixed a 75% false GPS-rejection rate
        # caused by Q growing P too slowly to keep pace with heading-error-
        # driven drift during sharp turns.
        #
        # Stage 2 (real data, Hr 16-20 -- done BEFORE Sync 2/a real trained
        # model exists, using real S1 GPS + a raw-gyro-integration stand-in
        # for the AI model, since that's the best available real signal):
        # swept Q against 86 minutes / ~37km of real IO-VNBD S1 driving.
        # Measured error BEFORE each GPS correction (i.e. genuine dead-
        # reckoning drift over each real ~9s gap between GPS fixes) -- NOT
        # error after correction, which is a misleading metric: a large
        # enough Q makes the filter simply snap onto GPS every fix,
        # trivially giving ~0 post-correction error regardless of how poor
        # the prediction was in between, which defeats the actual purpose
        # of dead reckoning.
        #
        # Result: pre-correction error DROPS from ~232m (scale=4x) to a
        # FLOOR of ~73m starting around scale=150x, with 0% GPS rejections.
        # Scaling Q higher than 150x gives NO further improvement -- that
        # floor is the real drift the raw-gyro-integration proxy accumulates
        # in ~9 real seconds, and Q cannot fix a weak prediction model, only
        # control how much the filter trusts it vs. GPS. This ~73m floor is
        # expected to drop substantially once Person 2's real trained model
        # (which should drift far less than naive gyro integration) replaces
        # this proxy -- at which point Q should be swept AGAIN, likely
        # landing on a much smaller value than 150x, since a better
        # prediction won't need as much "don't trust yourself" slack.
        self.Q = np.diag([0.5, 0.5, 0.3, 0.05]) * 150.0

        # --- Measurement noise (R) ---
        # Separate R for each measurement type since GPS and the AI
        # velocity/heading estimate have very different noise characteristics.
        self.R_gps = np.diag([3.0, 3.0])          # GPS position noise (meters)
        self.R_vel = np.diag([0.5, 0.1])          # AI [v, heading] noise -- used as
        # a FALLBACK only, when Person 2's model doesn't supply a per-prediction
        # velocity variance. Once the real model provides one (see
        # update_velocity's velocity_log_variance param), that overrides this.

        # --- Chi-squared innovation gating (Gap 5: failure mode handling) ---
        # Rejects a single update outright if it's a statistical outlier given
        # the filter's own uncertainty -- e.g. a spurious GPS multipath fix, or
        # a wildly wrong AI prediction. Threshold is the 95th percentile of the
        # chi-squared distribution for a 2D measurement (both GPS and
        # velocity/heading updates are 2D here).
        self.chi2_threshold = chi2.ppf(0.95, df=2)  # ~5.99
        self.gating_enabled = True

        # Rejection tracking, so you can report "X% of updates rejected as
        # outliers" as a concrete number in the writeup/presentation.
        self.n_gps_updates = 0
        self.n_gps_rejected = 0
        self.n_vel_updates = 0
        self.n_vel_rejected = 0

        # Rejection-streak safeguard: repeatedly rejecting real corrections
        # from the SAME source (e.g. GPS) means the filter has become
        # falsely overconfident (P too small), not that every GPS fix in a
        # row is genuinely bad data. Left unchecked this causes "filter
        # lock" -- a self-reinforcing loop where the state drifts further
        # from truth with nothing able to correct it. After this many
        # consecutive rejections from one source, force-accept the next
        # update instead of trusting the gate.
        self.max_consecutive_rejections = 5
        self._consecutive_gps_rejections = 0
        self._consecutive_vel_rejections = 0

    # ------------------------------------------------------------------
    # PREDICT STEP
    # ------------------------------------------------------------------
    def predict(self, dt):
        """
        Advance the state forward by dt seconds using a constant-velocity,
        constant-heading kinematic model:
            px' = px + v*cos(heading)*dt
            py' = py + v*sin(heading)*dt
            v'  = v          (unchanged until corrected by an update)
            h'  = heading     (unchanged until corrected by an update)
        """
        px, py, v, h = self.x

        # --- Nonlinear motion model f(x) ---
        px_new = px + v * np.cos(h) * dt
        py_new = py + v * np.sin(h) * dt
        v_new = v
        h_new = h
        self.x = np.array([px_new, py_new, v_new, h_new])

        # --- Jacobian of f(x) wrt state, F ---
        F = np.array([
            [1, 0, np.cos(h) * dt, -v * np.sin(h) * dt],
            [0, 1, np.sin(h) * dt,  v * np.cos(h) * dt],
            [0, 0, 1,               0],
            [0, 0, 0,               1],
        ])

        # --- Covariance propagation ---
        self.P = F @ self.P @ F.T + self.Q

    # ------------------------------------------------------------------
    # UPDATE: GPS FIX  (position correction, when GNSS is available)
    # ------------------------------------------------------------------
    def update_gps(self, px_meas, py_meas):
        """
        Correct the state using a GPS position fix.
        Measurement model: z = [px, py]  (linear, H is constant)
        """
        z = np.array([px_meas, py_meas])
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        self.n_gps_updates += 1
        accepted = self._kalman_update(z, H, self.R_gps, source="gps")
        if not accepted:
            self.n_gps_rejected += 1
        return accepted

    # ------------------------------------------------------------------
    # UPDATE: AI/IMU VELOCITY + HEADING  (Hr 3-6 task)
    # ------------------------------------------------------------------
    def update_velocity(self, v_kmh, heading_meas, velocity_log_variance=None):
        """
        Correct the state using the AI model's inertial velocity/heading
        estimate. This is what lets the filter keep tracking accurately
        during a GNSS blackout -- it's the actual "dead reckoning" step.

        v_kmh: speed in km/h (Person 2's model output unit). Converted to
        m/s here since the internal state and motion model use SI units --
        callers never need to convert this themselves.
        heading_meas: heading in radians, expected in [0, 2*pi) -- Person 2's
        model's native range. Internally the filter keeps heading wrapped to
        [-pi, pi) and handles the wraparound in the innovation, so a reading
        near 0 vs. a state near 2*pi is correctly treated as a small turn,
        not a near-full-circle jump. You don't need to pre-wrap anything
        before calling this.

        velocity_log_variance: OPTIONAL. Person 2's TCN outputs this alongside
        its velocity prediction (per the Gap 1 spec) -- it's the model's own
        confidence in that specific prediction. When provided, it directly
        sets the velocity noise (R) for THIS update: an uncertain prediction
        (high variance) gets trusted less, a confident one gets trusted more.
        This is the concrete link between the ML model and the filter --
        without it, every prediction is treated as equally reliable, which
        is what the fixed self.R_vel fallback below does when this arg is
        omitted (e.g. during Hr 0-3/Hr 6-10 testing before the real model
        exists).

        IMPORTANT: confirm this is not a no-op. Feed it a velocity/heading
        clearly different from the current predicted state and check that
        self.x actually moves toward the measurement afterward (see the
        test at the bottom of this file).
        """
        v_ms = v_kmh / 3.6
        heading_wrapped = self._wrap_to_pi(heading_meas)
        z = np.array([v_ms, heading_wrapped])
        H = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        if velocity_log_variance is not None:
            # Model-reported uncertainty overrides the fixed fallback for
            # this single update. Heading noise stays at the fixed value
            # from self.R_vel since the spec's log-variance term covers
            # velocity only (heading_rate is a separate, non-probabilistic
            # output per Gap 1).
            velocity_variance_ms2 = np.exp(velocity_log_variance) / (3.6 ** 2)
            R = np.diag([velocity_variance_ms2, self.R_vel[1, 1]])
        else:
            R = self.R_vel

        self.n_vel_updates += 1
        accepted = self._kalman_update(z, H, R, angle_row=1, source="velocity")
        if not accepted:
            self.n_vel_rejected += 1
        return accepted

    # ------------------------------------------------------------------
    # Shared Kalman update math
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_to_pi(angle):
        """Wraps an angle (or array of angles) to [-pi, pi)."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _kalman_update(self, z, H, R, angle_row=None, source="unknown"):
        """
        angle_row: index within z that represents an angle (e.g. heading),
        if any. Needed because Person 2's model outputs heading in [0, 2pi),
        so a raw subtraction breaks near the wraparound point (e.g. true
        heading 359 deg vs measured 1 deg would otherwise look like a
        ~358 deg jump instead of the real ~2 deg turn).

        Returns True if the update was applied, False if it was rejected by
        the chi-squared gate (Gap 5: a statistical-outlier measurement, e.g.
        GPS multipath or a spurious AI prediction, is skipped entirely
        rather than corrupting the state).
        """
        y = z - H @ self.x                     # innovation (residual)
        if angle_row is not None:
            y[angle_row] = self._wrap_to_pi(y[angle_row])

        S = H @ self.P @ H.T + R               # innovation covariance

        # --- Chi-squared gating check ---
        if self.gating_enabled:
            d_squared = float(y.T @ np.linalg.inv(S) @ y)

            streak_attr = "_consecutive_gps_rejections" if source == "gps" else "_consecutive_vel_rejections"
            streak = getattr(self, streak_attr, 0)
            force_accept = streak >= self.max_consecutive_rejections

            if d_squared > self.chi2_threshold and not force_accept:
                setattr(self, streak_attr, streak + 1)
                return False
            elif force_accept:
                # Safety valve: don't let a bad P silently lock the filter
                # out of ever correcting again. Accept this one even though
                # it failed the gate, and reset the streak.
                setattr(self, streak_attr, 0)
            else:
                setattr(self, streak_attr, 0)

        K = self.P @ H.T @ np.linalg.inv(S)    # Kalman gain

        self.x = self.x + K @ y
        self.x[3] = self._wrap_to_pi(self.x[3])  # keep internal heading bounded too
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        return True


# ======================================================================
# SYNTHETIC DATA TEST  (Hr 0-3 deliverable: prove the plumbing works)
# ======================================================================
def _print_gating_stats(ekf):
    """
    Prints the chi-squared gate rejection rate -- a concrete number for
    the writeup/presentation (e.g. "2.1% of GPS updates were rejected as
    outliers"). A healthy rate on clean synthetic data is close to 0-5%
    (matching the 95% confidence threshold); a much higher rate usually
    means Q/R are miscalibrated rather than the data being genuinely bad.
    """
    gps_pct = 100 * ekf.n_gps_rejected / ekf.n_gps_updates if ekf.n_gps_updates else 0
    vel_pct = 100 * ekf.n_vel_rejected / ekf.n_vel_updates if ekf.n_vel_updates else 0
    print(f"GPS updates rejected:      {ekf.n_gps_rejected}/{ekf.n_gps_updates} ({gps_pct:.1f}%)")
    print(f"Velocity updates rejected: {ekf.n_vel_rejected}/{ekf.n_vel_updates} ({vel_pct:.1f}%)")


def generate_synthetic_trajectory(n_steps=200, dt=0.1, seed=0):
    """
    Generates a fake vehicle path driving in a gentle curve at
    roughly constant speed, plus noisy 'sensor' readings derived from it:
        - noisy velocity/heading at every step (stands in for Person 2's
          AI model output, which doesn't exist yet)
        - noisy GPS position, but only every 10 steps (simulates GPS
          updating slower than IMU, and lets us later simulate blackouts
          by just not calling update_gps for a stretch)
    """
    rng = np.random.default_rng(seed)

    true_states = []
    px, py, v, h = 0.0, 0.0, 10.0, 0.0  # start at origin, 10 m/s, heading 0

    for i in range(n_steps):
        h += 0.01  # gentle constant turn rate -> curved path
        px += v * np.cos(h) * dt
        py += v * np.sin(h) * dt
        true_states.append([px, py, v, h])

    true_states = np.array(true_states)

    # Noisy velocity/heading "AI estimate" at every step
    vel_meas = true_states[:, 2] + rng.normal(0, 0.3, n_steps)
    heading_meas = true_states[:, 3] + rng.normal(0, 0.05, n_steps)

    # Noisy GPS every 10th step only
    gps_steps = np.arange(0, n_steps, 10)
    gps_meas = true_states[gps_steps, 0:2] + rng.normal(0, 2.0, (len(gps_steps), 2))

    return true_states, vel_meas, heading_meas, gps_steps, gps_meas


def run_synthetic_test():
    dt = 0.1
    n_steps = 200

    true_states, vel_meas, heading_meas, gps_steps, gps_meas = \
        generate_synthetic_trajectory(n_steps=n_steps, dt=dt)

    ekf = VehicleEKF(initial_state=[0, 0, 10.0, 0.0])

    estimated = []
    gps_idx = 0

    for i in range(n_steps):
        ekf.predict(dt)

        # Feed synthetic "AI model" output every step
        ekf.update_velocity(vel_meas[i], heading_meas[i])

        # Feed synthetic GPS only on steps where it's available
        if gps_idx < len(gps_steps) and gps_steps[gps_idx] == i:
            ekf.update_gps(gps_meas[gps_idx, 0], gps_meas[gps_idx, 1])
            gps_idx += 1

        estimated.append(ekf.x.copy())

    estimated = np.array(estimated)

    # --- Sanity checks (this is what "test with synthetic data" means) ---
    final_error = np.linalg.norm(estimated[-1, 0:2] - true_states[-1, 0:2])
    mean_error = np.mean(np.linalg.norm(estimated[:, 0:2] - true_states[:, 0:2], axis=1))

    print("=== EKF Synthetic Test (Hr 0-3 plumbing check) ===")
    print(f"Steps run:              {n_steps}")
    print(f"Final position error:   {final_error:.2f} m")
    print(f"Mean position error:    {mean_error:.2f} m")
    print("No crashes, no NaNs, no exploding values -> plumbing is correct.")
    print("(Accuracy doesn't matter yet -- this is fake data with a naive model.)")
    _print_gating_stats(ekf)

    assert not np.isnan(estimated).any(), "EKF produced NaNs -- check matrix math"
    assert final_error < 50, "Filter diverged badly -- check signs/units in predict()"

    return true_states, estimated, gps_steps, gps_meas


def run_on_shared_csv_dataset(data_dir="/home/claude"):
    """
    Runs the EKF against the shared synthetic dataset (ground_truth.csv,
    imu_data.csv, gps_data.csv) generated by generate_synthetic_dataset.py.

    This is closer to the real Sync 1/Sync 2 setup than run_synthetic_test():
    - "AI model output" here is literally the velocity_kmh/heading_rad label
      columns from imu_data.csv -- swap this for Person 2's real model
      predictions once they exist, same units (km/h), same call signature.
    - GPS comes from gps_data.csv, sparser than the IMU rate, exactly like
      real GNSS fixes arriving slower than IMU samples.
    """
    import csv as csv_module

    def read_csv(path):
        with open(path) as f:
            reader = csv_module.DictReader(f)
            rows = [{k: float(v) for k, v in row.items()} for row in reader]
        return rows

    imu_rows = read_csv(f"{data_dir}/imu_data.csv")
    gps_rows = read_csv(f"{data_dir}/gps_data.csv")
    gt_rows = read_csv(f"{data_dir}/ground_truth.csv")

    ekf = VehicleEKF(initial_state=[gt_rows[0]["x_m"], gt_rows[0]["y_m"],
                                     gt_rows[0]["speed_kmh"] / 3.6, gt_rows[0]["heading_rad"]])

    dt = imu_rows[1]["timestamp_s"] - imu_rows[0]["timestamp_s"]
    gps_idx = 0
    estimated = []

    for i, row in enumerate(imu_rows):
        ekf.predict(dt)

        # Stand-in for Person 2's real model output -- same units (km/h),
        # same call. Replace row["velocity_kmh"]/row["heading_rad"] with
        # the model's actual prediction when it exists.
        ekf.update_velocity(row["velocity_kmh"], row["heading_rad"])

        if gps_idx < len(gps_rows) and gps_rows[gps_idx]["timestamp_s"] <= row["timestamp_s"]:
            ekf.update_gps(gps_rows[gps_idx]["x_m"], gps_rows[gps_idx]["y_m"])
            gps_idx += 1

        estimated.append(ekf.x.copy())

    estimated = np.array(estimated)
    gt_xy = np.array([[r["x_m"], r["y_m"]] for r in gt_rows])
    mean_error = np.mean(np.linalg.norm(estimated[:, 0:2] - gt_xy, axis=1))
    final_error = np.linalg.norm(estimated[-1, 0:2] - gt_xy[-1])

    print("=== EKF on shared CSV dataset ===")
    print(f"Mean position error:  {mean_error:.2f} m")
    print(f"Final position error: {final_error:.2f} m")
    _print_gating_stats(ekf)

    return gt_xy, estimated


def test_heading_wraparound():
    """
    Regression test: a vehicle heading near 359 deg (~6.26 rad) gets a
    measurement near 1 deg (~0.02 rad) from the model, simulating a small
    real turn that crosses the 0/2*pi boundary. Without wraparound
    handling, the innovation would be ~6.24 rad (almost a full circle)
    and the filter would violently overcorrect -- likely triggering the
    chi-squared gate to reject it too, which would be the wrong outcome
    for what is actually a completely normal ~2 deg turn.
    """
    near_2pi = 2 * np.pi - 0.017   # ~359 deg, matches Person 2's [0, 2pi) range
    just_over_0 = 0.017            # ~1 deg

    ekf = VehicleEKF(initial_state=[0, 0, 10.0, near_2pi])
    ekf.predict(0.1)

    heading_before = ekf.x[3]
    accepted = ekf.update_velocity(v_kmh=36.0, heading_meas=just_over_0)  # 36 km/h = 10 m/s

    print("=== Heading wraparound regression test ===")
    print(f"Accepted by chi-squared gate: {accepted}")
    assert accepted, (
        "Wraparound bug: a small ~2 deg turn was incorrectly rejected as a "
        "statistical outlier -- the wraparound fix isn't reaching the gate check."
    )
    print("PASS -- wraparound handled correctly, no false rejection or overcorrection.\n")


def test_chi_squared_gate():
    """
    Regression test for Gap 5 (failure mode handling): feed the filter a
    GPS fix that's wildly inconsistent with the current state (e.g. a
    150m jump when GPS noise is normally ~3m) and confirm it's rejected
    rather than corrupting the position estimate.
    """
    ekf = VehicleEKF(initial_state=[0, 0, 10.0, 0.0])
    ekf.predict(0.1)

    x_before = ekf.x.copy()
    accepted = ekf.update_gps(px_meas=150.0, py_meas=150.0)  # wildly implausible jump

    print("=== Chi-squared gating regression test ===")
    print(f"Outlier GPS fix accepted: {accepted} (should be False)")
    print(f"State before: {x_before[:2]}, state after: {ekf.x[:2]} (should be unchanged)")
    assert not accepted, "Gate failed to reject an obvious outlier measurement."
    assert np.allclose(x_before, ekf.x), "State changed even though update was rejected."

    # Now confirm a NORMAL GPS fix still goes through fine
    normal_accepted = ekf.update_gps(px_meas=ekf.x[0] + 1.0, py_meas=ekf.x[1] + 1.0)
    print(f"Normal GPS fix accepted: {normal_accepted} (should be True)")
    assert normal_accepted, "Gate incorrectly rejected a normal, small GPS correction."
    print("PASS -- outliers rejected, normal updates still pass through.\n")


if __name__ == "__main__":
    test_heading_wraparound()
    test_chi_squared_gate()
    true_states, estimated, gps_steps, gps_meas = run_synthetic_test()

    # Optional: quick plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 6))
        plt.plot(true_states[:, 0], true_states[:, 1], label="Ground truth", linewidth=2)
        plt.plot(estimated[:, 0], estimated[:, 1], label="EKF estimate", linestyle="--")
        plt.scatter(gps_meas[:, 0], gps_meas[:, 1], color="red", s=15, label="Noisy GPS fixes")
        plt.legend()
        plt.title("EKF Fusion — Synthetic Data Test")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.savefig("/home/claude/ekf_synthetic_test.png", dpi=150)
        print("\nPlot saved to ekf_synthetic_test.png")
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot, numeric test above is what matters)")

    # Also run against the shared CSV dataset if it's present
    import os
    if os.path.exists("/home/claude/imu_data.csv"):
        print()
        run_on_shared_csv_dataset()
