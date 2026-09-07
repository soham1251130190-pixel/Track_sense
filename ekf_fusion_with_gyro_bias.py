"""
ekf_fusion_with_gyro_bias.py
------------------------------
Experimental variant of VehicleEKF adding gyro_bias as an estimated state,
per the Gap 3 spec's full 6-state design (simplified here to 5-state: we
skip separate vx/vy since our GPS/velocity updates don't need that split,
but DO add gyro_bias, which is what the real-data test showed actually
matters -- see fusion_spec_final.md's "Known Limitations" section).

WHY THIS IS A SEPARATE FILE, NOT AN EDIT TO ekf_fusion.py:
ekf_fusion.py is already tested and documented. This changes predict()'s
signature (now needs a raw gyro reading, not just dt) which is a breaking
API change -- kept isolated here until proven better, so it doesn't risk
destabilizing already-working, already-shared code.

STATE VECTOR CHANGE:
    Old (ekf_fusion.py):  x = [px, py, v, heading]              (4-state)
    New (this file):      x = [px, py, v, heading, gyro_bias]   (5-state)

WHY THIS SHOULD HELP:
The real-data test (run_ekf_on_real_data.py) showed a ~73m drift floor
per ~9s GPS gap, using raw gyro_yaw integration as a stand-in for Person
2's model. That drift is exactly what a persistent, slowly-varying gyro
bias causes: MEMS gyroscopes have a small but non-zero offset even when
stationary, and naive integration (heading += gyro_yaw*dt) never corrects
for it, so error compounds linearly with every integration step over the
whole 86-minute drive.

By making gyro_bias part of the filter's state, GPS corrections to
heading can ALSO correct the estimated bias (via the covariance coupling
built during predict()), letting the filter learn and cancel out the
bias over time instead of accumulating it forever.
"""

import numpy as np
from scipy.stats import chi2


class VehicleEKFWithGyroBias:
    def __init__(self, initial_state=None, initial_covariance=None):
        """
        initial_state: array-like [px, py, v, heading, gyro_bias].
        Defaults to zeros, with gyro_bias starting at 0 (no known bias yet
        -- the filter will estimate it as GPS corrections come in).
        """
        self.n = 5  # state dimension (was 4 in ekf_fusion.py)

        if initial_state is None:
            initial_state = np.zeros(self.n)
        self.x = np.array(initial_state, dtype=float).reshape(self.n)

        if initial_covariance is None:
            # Extra uncertainty term for gyro_bias -- we genuinely don't
            # know it at the start, so give it a reasonably loose prior.
            initial_covariance = np.diag([5.0, 5.0, 2.0, 0.5, 0.1])
        self.P = np.array(initial_covariance, dtype=float)

        # Process noise -- same tuned values as ekf_fusion.py for the first
        # 4 states, PLUS a new term for gyro_bias. Bias should drift very
        # slowly (it's a near-constant sensor property, not something that
        # jumps around), so this is intentionally small.
        self.Q = np.diag([0.5, 0.5, 0.3, 0.05, 0.001]) * 150.0

        self.R_gps = np.diag([3.0, 3.0])
        self.R_vel = np.diag([0.5, 0.1])

        self.chi2_threshold = chi2.ppf(0.95, df=2)
        self.gating_enabled = True

        self.n_gps_updates = 0
        self.n_gps_rejected = 0
        self.n_vel_updates = 0
        self.n_vel_rejected = 0
        self.max_consecutive_rejections = 5
        self._consecutive_gps_rejections = 0
        self._consecutive_vel_rejections = 0

    # ------------------------------------------------------------------
    # PREDICT STEP -- now takes a raw gyro_z reading, not just dt
    # ------------------------------------------------------------------
    def predict(self, dt, gyro_z):
        """
        gyro_z: raw gyroscope yaw-rate reading (rad/s), BEFORE bias
        correction -- this is the actual sensor value, not an AI estimate.
        The filter subtracts its own current bias estimate internally:
            heading_new = heading + (gyro_z - gyro_bias) * dt
        This is the mechanism that lets gyro_bias get estimated: as GPS
        corrects heading over time, the Kalman gain also nudges gyro_bias
        (via the F-matrix coupling below), gradually learning the true
        sensor offset instead of letting it accumulate unchecked forever.
        """
        px, py, v, h, bias = self.x

        corrected_rate = gyro_z - bias
        h_new = h + corrected_rate * dt
        px_new = px + v * np.cos(h) * dt
        py_new = py + v * np.sin(h) * dt
        v_new = v
        bias_new = bias  # assumed constant between updates; Q lets it drift slowly

        self.x = np.array([px_new, py_new, v_new, h_new, bias_new])

        # Jacobian -- same as ekf_fusion.py's 4x4 block, PLUS:
        #   - a new column for d(heading_new)/d(gyro_bias) = -dt
        #     (this is THE key coupling that makes bias estimation work)
        #   - a new row for gyro_bias's own (trivial, identity) dynamics
        F = np.array([
            [1, 0, np.cos(h) * dt, -v * np.sin(h) * dt, 0],
            [0, 1, np.sin(h) * dt,  v * np.cos(h) * dt, 0],
            [0, 0, 1,               0,                  0],
            [0, 0, 0,               1,                 -dt],
            [0, 0, 0,               0,                  1],
        ])

        self.P = F @ self.P @ F.T + self.Q

    # ------------------------------------------------------------------
    # UPDATE: GPS FIX -- same as ekf_fusion.py, just extended to 5-state
    # ------------------------------------------------------------------
    def update_gps(self, px_meas, py_meas):
        z = np.array([px_meas, py_meas])
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ])
        self.n_gps_updates += 1
        accepted = self._kalman_update(z, H, self.R_gps, source="gps")
        if not accepted:
            self.n_gps_rejected += 1
        return accepted

    # ------------------------------------------------------------------
    # UPDATE: AI/IMU VELOCITY + HEADING -- same as ekf_fusion.py
    # ------------------------------------------------------------------
    def update_velocity(self, v_kmh, heading_meas, velocity_log_variance=None):
        v_ms = v_kmh / 3.6
        heading_wrapped = self._wrap_to_pi(heading_meas)
        z = np.array([v_ms, heading_wrapped])
        H = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ])

        if velocity_log_variance is not None:
            velocity_variance_ms2 = np.exp(velocity_log_variance) / (3.6 ** 2)
            R = np.diag([velocity_variance_ms2, self.R_vel[1, 1]])
        else:
            R = self.R_vel

        self.n_vel_updates += 1
        accepted = self._kalman_update(z, H, R, angle_row=1, source="velocity")
        if not accepted:
            self.n_vel_rejected += 1
        return accepted

    @staticmethod
    def _wrap_to_pi(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _kalman_update(self, z, H, R, angle_row=None, source="unknown"):
        y = z - H @ self.x
        if angle_row is not None:
            y[angle_row] = self._wrap_to_pi(y[angle_row])

        S = H @ self.P @ H.T + R

        if self.gating_enabled:
            d_squared = float(y.T @ np.linalg.inv(S) @ y)
            streak_attr = "_consecutive_gps_rejections" if source == "gps" else "_consecutive_vel_rejections"
            streak = getattr(self, streak_attr, 0)
            force_accept = streak >= self.max_consecutive_rejections

            if d_squared > self.chi2_threshold and not force_accept:
                setattr(self, streak_attr, streak + 1)
                return False
            elif force_accept:
                setattr(self, streak_attr, 0)
            else:
                setattr(self, streak_attr, 0)

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[3] = self._wrap_to_pi(self.x[3])  # wrap heading
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        return True
