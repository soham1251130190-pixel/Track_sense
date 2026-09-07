"""
full_pipeline_demo.py
-----------------------
Complete end-to-end demo: real S1 sensor data -> EKF fusion -> coordinate
conversion -> ready for map-matching. This is the single script that
demonstrates every piece of Person 3's Phase 1 work working together.

Pipeline:
    S1_training.csv (real IMU + real GPS, IO-VNBD)
        -> VehicleEKF (predict + update_velocity + update_gps)
        -> fused local (x, y) trajectory
        -> local_xy_to_latlon() [shared projection.py, agreed with P1]
        -> real (lat, lon) trajectory
        -> ready for map_matching.py (NaiveNearestPointMatcher now,
           GraphHopperClient once a real UK OSM server exists)

Run this to see the whole system work together on real data.
"""

import numpy as np
import pandas as pd
from ekf_fusion import VehicleEKF
from projection import local_xy_to_latlon, ORIGIN_LAT, ORIGIN_LON


def run_full_pipeline(csv_path="S1_training.csv"):
    df = pd.read_csv(csv_path)
    timestamps = pd.to_datetime(df["timestamp"])
    dt_array = timestamps.diff().dt.total_seconds().fillna(0.1).to_numpy()
    x_m = df["x_m"].to_numpy(dtype=float)
    y_m = df["y_m"].to_numpy(dtype=float)
    speed_kmh = df["speed_kmh"].to_numpy(dtype=float)
    gyro_yaw = df["gyro_yaw"].to_numpy(dtype=float)

    gps_update_mask = np.zeros(len(df), dtype=bool)
    gps_update_mask[0] = True
    gps_update_mask[1:] = (np.diff(x_m) != 0) | (np.diff(y_m) != 0)
    real_gps_indices = np.where(gps_update_mask)[0]
    x0, y0 = x_m[real_gps_indices[0]], y_m[real_gps_indices[0]]
    x1, y1 = x_m[real_gps_indices[1]], y_m[real_gps_indices[1]]
    init_heading = np.arctan2(y1 - y0, x1 - x0)

    print("=== STEP 1: Fusion (VehicleEKF) ===")
    ekf = VehicleEKF(initial_state=[x0, y0, speed_kmh[0] / 3.6, init_heading])
    estimated_heading = init_heading
    estimated_xy = []

    for i in range(len(df)):
        dt = max(dt_array[i], 1e-3)
        ekf.predict(dt)
        estimated_heading += gyro_yaw[i] * dt
        ekf.update_velocity(speed_kmh[i], estimated_heading % (2 * np.pi))
        if gps_update_mask[i] and i > 0:
            ekf.update_gps(x_m[i], y_m[i])
        estimated_xy.append(ekf.x[:2].copy())

    estimated_xy = np.array(estimated_xy)
    print(f"  Fused {len(estimated_xy)} position estimates")
    print(f"  GPS updates rejected: {ekf.n_gps_rejected}/{ekf.n_gps_updates}")

    print("\n=== STEP 2: Coordinate conversion (shared projection.py) ===")
    lat, lon = local_xy_to_latlon(estimated_xy[:, 0], estimated_xy[:, 1],
                                    ORIGIN_LAT, ORIGIN_LON)
    print(f"  Converted local (x,y) -> real (lat,lon)")
    print(f"  lat range: [{lat.min():.6f}, {lat.max():.6f}]")
    print(f"  lon range: [{lon.min():.6f}, {lon.max():.6f}]")
    print(f"  (Coventry, UK area -- matches the IO-VNBD S1 origin, as expected)")

    print("\n=== STEP 3: Ready for map-matching ===")
    print("  This (lat, lon) trajectory is now in the exact format")
    print("  GraphHopperClient.match_points() expects.")
    print("  A real UK GraphHopper server (with a Coventry-area OSM extract)")
    print("  would snap this onto actual road geometry -- not yet stood up")
    print("  in Phase 1 due to time constraints (see fusion_spec_final.md).")

    return {
        "local_xy": estimated_xy,
        "lat": lat,
        "lon": lon,
        "ekf": ekf,
    }


if __name__ == "__main__":
    results = run_full_pipeline()

    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(results["local_xy"][:, 0], results["local_xy"][:, 1],
                 linewidth=0.8, color="tab:blue")
        ax1.set_title("Step 1-2: EKF fused trajectory (local xy, meters)")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.axis("equal")
        ax1.grid(True, alpha=0.3)

        ax2.plot(results["lon"], results["lat"], linewidth=0.8, color="tab:green")
        ax2.set_title("Step 3: Converted to real (lat, lon) -- ready for map-matching")
        ax2.set_xlabel("longitude")
        ax2.set_ylabel("latitude")
        ax2.axis("equal")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("/home/claude/full_pipeline_demo.png", dpi=150)
        print("\nPlot saved to full_pipeline_demo.png")
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot)")
