"""
generate_synthetic_dataset.py
------------------------------
Shared synthetic dataset for Person 2 (ML Model) and Person 3 (Fusion Engineer).

Why shared: if you each invent your own fake data, your Sync 1 "walking
skeleton" test proves nothing -- your EKF might silently expect different
units/format than what Person 2's dummy model actually outputs. Generating
one ground-truth trajectory and deriving both your inputs from it means
Sync 1 actually tests the real interface.

Produces 3 CSV files from one fake vehicle drive:

1. ground_truth.csv   - the "answer key". Timestamp, position (x,y in meters,
                         local flat-earth frame), speed (km/h), heading (rad).
                         Nobody trains/filters on this directly -- it's only
                         for scoring (Person 4's eval harness will want this too).

2. imu_data.csv       - PERSON 2 INPUT. High-rate (50 Hz) noisy accelerometer
                         (accel_x, accel_y, m/s^2) and gyroscope (gyro_z, rad/s)
                         readings, as if from SensorManager. This stands in for
                         real IMU data until the IO-VNBD subset is loaded.
                         Label columns (velocity_kmh, heading_rad) are included
                         for supervised training -- drop them for real inference.

3. gps_data.csv       - PERSON 3 INPUT. Sparse (1 Hz) noisy GPS fixes
                         (x, y in meters, same local frame as ground truth).
                         Feed these into VehicleEKF.update_gps().

UNITS NOTE: velocity is in km/h everywhere in these files and in the AI
model's expected output, matching what was specified. The EKF internally
converts km/h -> m/s because the kinematic motion model needs SI units --
see the updated ekf_fusion.py, you don't need to convert anything yourself
before calling update_velocity(v_kmh, heading_rad).
"""

import numpy as np
import csv


def generate_shared_dataset(
    duration_s=60.0,
    imu_rate_hz=50.0,
    gps_rate_hz=1.0,
    seed=0,
    out_dir="/home/claude",
):
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Generate ground-truth trajectory at IMU rate (finest resolution)
    # ------------------------------------------------------------------
    dt = 1.0 / imu_rate_hz
    n_steps = int(duration_s * imu_rate_hz)
    t = np.arange(n_steps) * dt

    px, py, heading = 0.0, 0.0, 0.0
    speed_ms = 12.0  # start at 12 m/s (~43 km/h)

    gt_x, gt_y, gt_speed_ms, gt_heading = [], [], [], []

    for i in range(n_steps):
        # Fake but plausible driving pattern: gentle speed variation +
        # a few turns, so it's not just a straight line at constant speed
        speed_ms += rng.normal(0, 0.05)                     # gentle accel noise
        speed_ms = np.clip(speed_ms, 3.0, 25.0)              # keep within 11-90 km/h
        turn_rate = 0.15 * np.sin(t[i] / 8.0)                # smooth wandering turns
        heading += turn_rate * dt

        px += speed_ms * np.cos(heading) * dt
        py += speed_ms * np.sin(heading) * dt

        gt_x.append(px)
        gt_y.append(py)
        gt_speed_ms.append(speed_ms)
        gt_heading.append(heading)

    gt_x = np.array(gt_x)
    gt_y = np.array(gt_y)
    gt_speed_ms = np.array(gt_speed_ms)
    gt_heading = np.array(gt_heading)
    gt_speed_kmh = gt_speed_ms * 3.6

    # ------------------------------------------------------------------
    # 2. Derive noisy IMU stream (accel_x, accel_y, gyro_z) from ground truth
    # ------------------------------------------------------------------
    # Acceleration = derivative of velocity vector; gyro_z = derivative of heading
    vx = gt_speed_ms * np.cos(gt_heading)
    vy = gt_speed_ms * np.sin(gt_heading)

    accel_x = np.gradient(vx, dt)
    accel_y = np.gradient(vy, dt)
    gyro_z = np.gradient(gt_heading, dt)

    # Add sensor noise + a small constant bias per axis, since real IMUs drift
    accel_x_noisy = accel_x + rng.normal(0, 0.4, n_steps) + 0.05
    accel_y_noisy = accel_y + rng.normal(0, 0.4, n_steps) - 0.03
    gyro_z_noisy = gyro_z + rng.normal(0, 0.02, n_steps) + 0.005

    # Occasionally inject a pothole/vibration burst, since real roads aren't smooth
    n_bumps = int(duration_s / 4)
    bump_indices = rng.choice(n_steps, size=n_bumps, replace=False)
    for idx in bump_indices:
        accel_x_noisy[idx] += rng.normal(0, 2.5)
        accel_y_noisy[idx] += rng.normal(0, 2.5)

    # ------------------------------------------------------------------
    # 3. Derive sparse noisy GPS stream
    # ------------------------------------------------------------------
    gps_step = int(imu_rate_hz / gps_rate_hz)
    gps_indices = np.arange(0, n_steps, gps_step)
    gps_x = gt_x[gps_indices] + rng.normal(0, 3.0, len(gps_indices))
    gps_y = gt_y[gps_indices] + rng.normal(0, 3.0, len(gps_indices))
    gps_t = t[gps_indices]

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    with open(f"{out_dir}/ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "x_m", "y_m", "speed_kmh", "heading_rad"])
        for i in range(n_steps):
            w.writerow([f"{t[i]:.3f}", f"{gt_x[i]:.4f}", f"{gt_y[i]:.4f}",
                        f"{gt_speed_kmh[i]:.3f}", f"{gt_heading[i]:.5f}"])

    with open(f"{out_dir}/imu_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "accel_x_mps2", "accel_y_mps2", "gyro_z_radps",
                     "velocity_kmh", "heading_rad"])
        for i in range(n_steps):
            w.writerow([f"{t[i]:.3f}", f"{accel_x_noisy[i]:.4f}", f"{accel_y_noisy[i]:.4f}",
                        f"{gyro_z_noisy[i]:.5f}", f"{gt_speed_kmh[i]:.3f}", f"{gt_heading[i]:.5f}"])

    with open(f"{out_dir}/gps_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_s", "x_m", "y_m"])
        for i in range(len(gps_indices)):
            w.writerow([f"{gps_t[i]:.3f}", f"{gps_x[i]:.4f}", f"{gps_y[i]:.4f}"])

    print(f"Generated {n_steps} IMU samples @ {imu_rate_hz} Hz ({duration_s:.0f}s drive)")
    print(f"Generated {len(gps_indices)} GPS fixes @ {gps_rate_hz} Hz")
    print(f"Speed range: {gt_speed_kmh.min():.1f} - {gt_speed_kmh.max():.1f} km/h")
    print(f"Files written: ground_truth.csv, imu_data.csv, gps_data.csv (in {out_dir})")

    return {
        "ground_truth": {"t": t, "x": gt_x, "y": gt_y, "speed_kmh": gt_speed_kmh, "heading": gt_heading},
        "imu": {"t": t, "accel_x": accel_x_noisy, "accel_y": accel_y_noisy, "gyro_z": gyro_z_noisy},
        "gps": {"t": gps_t, "x": gps_x, "y": gps_y},
    }


if __name__ == "__main__":
    generate_shared_dataset(out_dir="/home/claude")
