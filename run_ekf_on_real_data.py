"""
run_ekf_on_real_data.py
-------------------------
Runs the actual VehicleEKF (ekf_fusion.py) against the REAL IO-VNBD S1
dataset (S1_training.csv) -- not synthetic data.

IMPORTANT CAVEAT: Person 2's real trained model does not exist yet. So
"velocity/heading estimate" here is a STAND-IN, built directly from this
file's own raw sensor columns:
    - speed_kmh: used directly as the velocity input (this file already
      provides it, likely derived from wheel-speed/GPS in P1's pipeline --
      NOT a neural network prediction, so don't over-interpret results as
      "how good the AI model is." This only tests the EKF/gating logic
      against real-world noise characteristics and a real, aggressive
      driving pattern -- genuinely useful for that.
    - heading: NOT present as a column. Derived by integrating gyro_yaw
      (raw gyroscope yaw rate) over time: heading[k] = heading[k-1] +
      gyro_yaw*dt. This is exactly the kind of raw-to-corrected step
      Person 2's TCN is meant to learn to do BETTER than simple
      integration (which drifts) -- so this test's heading estimate
      will likely drift more than the real model eventually will.
      Initial heading is estimated from the direction between the first
      two real (non-forward-filled) GPS fixes.

GPS UPDATE HANDLING: x_m/y_m are forward-filled in the source file (see
load_real_gps_tracks.py's findings -- real updates happen only ~every 9s,
not every row). update_gps() is called ONLY at rows where x_m/y_m
actually change value, not all 51,746 rows -- calling it on every
repeated/duplicate value would look like a perfect, zero-noise GPS fix
firing at 10Hz, which is not what's really happening and would badly
distort the filter's gating behavior.
"""

import numpy as np
import pandas as pd
from ekf_fusion import VehicleEKF


def run_real_data_test(csv_path="S1_training.csv", max_rows=None):
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)

    timestamps = pd.to_datetime(df["timestamp"])
    dt_array = timestamps.diff().dt.total_seconds().fillna(0.1).to_numpy()

    x_m = df["x_m"].to_numpy(dtype=float)
    y_m = df["y_m"].to_numpy(dtype=float)
    speed_kmh = df["speed_kmh"].to_numpy(dtype=float)
    gyro_yaw = df["gyro_yaw"].to_numpy(dtype=float)

    # Identify rows where a REAL (non-forward-filled) GPS update occurred
    gps_update_mask = np.zeros(len(df), dtype=bool)
    gps_update_mask[0] = True  # treat the first row as a real fix (starting point)
    gps_update_mask[1:] = (np.diff(x_m) != 0) | (np.diff(y_m) != 0)
    n_real_gps_fixes = gps_update_mask.sum()

    # Estimate initial heading from the first real movement between fixes
    real_gps_indices = np.where(gps_update_mask)[0]
    x0, y0 = x_m[real_gps_indices[0]], y_m[real_gps_indices[0]]
    x1, y1 = x_m[real_gps_indices[1]], y_m[real_gps_indices[1]]
    init_heading = np.arctan2(y1 - y0, x1 - x0)
    init_heading_wrapped = init_heading % (2 * np.pi)  # match Person 2's [0, 2pi) convention

    print(f"=== Running EKF on REAL S1 data ({len(df)} rows) ===")
    print(f"Real GPS fixes found: {n_real_gps_fixes} (out of {len(df)} rows)")
    print(f"Initial heading estimate: {init_heading:.4f} rad ({np.degrees(init_heading):.1f} deg)")

    ekf = VehicleEKF(initial_state=[x0, y0, speed_kmh[0] / 3.6, init_heading])

    estimated_heading = init_heading  # running integrated heading (stand-in for AI model)
    estimated_positions = []

    for i in range(len(df)):
        dt = max(dt_array[i], 1e-3)  # guard against zero/negative dt
        ekf.predict(dt)

        # Integrate gyro_yaw as the stand-in "AI model" heading estimate
        estimated_heading += gyro_yaw[i] * dt
        heading_input = estimated_heading % (2 * np.pi)  # wrap into [0, 2pi) like real model output

        ekf.update_velocity(speed_kmh[i], heading_input)

        if gps_update_mask[i] and i > 0:  # skip i==0, already used as initial state
            ekf.update_gps(x_m[i], y_m[i])

        estimated_positions.append(ekf.x[:2].copy())

    estimated_positions = np.array(estimated_positions)

    # --- TWO different error metrics, deliberately kept separate ---
    #
    # 1. POST-correction error (state right after applying update_gps):
    #    misleading in isolation -- a large Q makes the filter simply snap
    #    onto GPS at every fix, making this trivially near-zero regardless
    #    of prediction quality. Included only for reference/debugging.
    #
    # 2. PRE-correction error (state right BEFORE applying update_gps, i.e.
    #    purely from predict() + update_velocity() since the last real GPS
    #    fix): this is the metric that actually matters -- it's the real
    #    dead-reckoning drift accumulated during each GPS gap, which is the
    #    literal problem this whole project exists to solve. This is what
    #    should be reported/compared when evaluating Q or a new model.
    post_correction_errors = []
    pre_correction_errors = []

    ekf2 = VehicleEKF(initial_state=[x0, y0, speed_kmh[0] / 3.6, init_heading])
    ekf2.Q = ekf.Q.copy()
    estimated_heading2 = init_heading
    for i in range(len(df)):
        dt = max(dt_array[i], 1e-3)
        ekf2.predict(dt)
        estimated_heading2 += gyro_yaw[i] * dt
        heading_input2 = estimated_heading2 % (2 * np.pi)
        ekf2.update_velocity(speed_kmh[i], heading_input2)

        if gps_update_mask[i] and i > 0:
            pre_err = np.linalg.norm(ekf2.x[:2] - np.array([x_m[i], y_m[i]]))
            pre_correction_errors.append(pre_err)
            ekf2.update_gps(x_m[i], y_m[i])
            post_err = np.linalg.norm(ekf2.x[:2] - np.array([x_m[i], y_m[i]]))
            post_correction_errors.append(post_err)

    pre_correction_errors = np.array(pre_correction_errors)
    post_correction_errors = np.array(post_correction_errors)

    print(f"\n[MISLEADING IN ISOLATION] Post-correction error (right after each GPS fix):")
    print(f"  Mean: {post_correction_errors.mean():.2f} m  "
          f"(near-zero here just means the filter trusts GPS a lot -- "
          f"NOT a measure of dead-reckoning quality)")

    print(f"\n[ACTUAL METRIC THAT MATTERS] Pre-correction error (dead-reckoning")
    print(f"drift accumulated since the last real GPS fix, ~9s gaps on average):")
    print(f"  Mean:   {pre_correction_errors.mean():.2f} m")
    print(f"  Median: {np.median(pre_correction_errors):.2f} m")
    print(f"  Max:    {pre_correction_errors.max():.2f} m")

    gps_pct = 100 * ekf.n_gps_rejected / ekf.n_gps_updates if ekf.n_gps_updates else 0
    vel_pct = 100 * ekf.n_vel_rejected / ekf.n_vel_updates if ekf.n_vel_updates else 0
    print(f"\nGPS updates rejected:      {ekf.n_gps_rejected}/{ekf.n_gps_updates} ({gps_pct:.1f}%)")
    print(f"Velocity updates rejected: {ekf.n_vel_rejected}/{ekf.n_vel_updates} ({vel_pct:.1f}%)")

    return {
        "true_xy": np.column_stack([x_m, y_m]),
        "estimated_xy": estimated_positions,
        "gps_update_mask": gps_update_mask,
        "errors_at_fixes": pre_correction_errors,
        "ekf": ekf,
    }


if __name__ == "__main__":
    results = run_real_data_test()

    try:
        import matplotlib.pyplot as plt

        true_xy = results["true_xy"]
        est_xy = results["estimated_xy"]
        mask = results["gps_update_mask"]

        plt.figure(figsize=(9, 8))
        plt.plot(true_xy[mask, 0], true_xy[mask, 1], "g-", linewidth=1.5,
                 label="Real GPS fixes (ground truth)", alpha=0.7)
        plt.plot(est_xy[:, 0], est_xy[:, 1], "orange", linewidth=0.8,
                 linestyle="--", label="EKF estimate", alpha=0.8)
        plt.scatter(true_xy[mask, 0], true_xy[mask, 1], color="green", s=8, zorder=5)
        plt.legend()
        plt.title("EKF on REAL IO-VNBD S1 Data (86 min, ~37 km)")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.axis("equal")
        plt.grid(True, alpha=0.3)
        plt.savefig("/home/claude/ekf_real_data_test.png", dpi=150)
        print("\nPlot saved to ekf_real_data_test.png")
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot)")
