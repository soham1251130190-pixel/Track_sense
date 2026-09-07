"""
load_real_gps_tracks.py
------------------------
Hr 6-10: loads Person 1's real IO-VNBD S1 dataset (S1_training.csv) and
prepares it for map-matching testing.

REAL FILE STRUCTURE (confirmed by inspecting the actual uploaded file --
this differs from what was originally described, see notes below):

    columns: timestamp, x_m, y_m, speed_kmh, accel_x, accel_y, accel_z,
             gyro_yaw, gyro_pitch, gyro_roll, gravity_x, gravity_y,
             gravity_z, mag_x, mag_y, mag_z

    - timestamp: a datetime STRING (e.g. "2019-09-08 10:07:49.546"),
      NOT a numeric offset in seconds like the synthetic CSVs used.
    - speed_kmh: NOT "speed" as originally described -- has the unit
      suffix. Matches the km/h convention already used elsewhere in
      this pipeline, so no unit conversion needed once loaded.
    - x_m/y_m only actually CHANGE roughly every ~9 seconds (median
      gap 9.0s, some gaps up to 70s) even though the file is sampled
      at 10Hz overall -- meaning GPS was forward-filled onto the full
      IMU-rate timeline by P1's pipeline, not interpolated, and updates
      far less often than the 1Hz stated in the IO-VNBD paper for this
      particular recording. Worth confirming with P1 whether this is
      expected for S1 specifically or a processing quirk.
    - Includes full raw IMU (accel/gyro/gravity/mag) -- useful for
      Person 2's training too, not just this map-matching test.
    - 51,746 rows, ~86 min of driving, ~37.2 km path, speeds up to
      ~18.8 km/h -- slow urban/residential driving, not highway.

Pipeline this script sets up:
    S1_training.csv (x_m, y_m from P1's pipeline, already in local xy
    using the shared IO-VNBD origin)
        -> convert to lat/lon using shared projection.py
        -> ready to feed into map_matching.py (NaiveNearestPointMatcher
           for now, GraphHopperClient once a real UK OSM server exists)
"""

import csv
import numpy as np
import pandas as pd
from projection import local_xy_to_latlon, ORIGIN_LAT, ORIGIN_LON


def load_gps_track(csv_path):
    """
    Reads the real S1_training.csv format:
    timestamp (datetime string), x_m, y_m, speed_kmh, plus raw IMU columns
    (accel_x/y/z, gyro_yaw/pitch/roll, gravity_x/y/z, mag_x/y/z) which are
    loaded but not used by this map-matching step -- they're there for
    Person 2's model training, not needed here.

    Returns a dict of numpy arrays: x, y, timestamp (as pandas datetime),
    speed_kmh, lat, lon.
    """
    df = pd.read_csv(csv_path)

    required_cols = {"timestamp", "x_m", "y_m", "speed_kmh"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}. "
            "Check with P1 whether the export format changed again."
        )

    x = df["x_m"].to_numpy(dtype=float)
    y = df["y_m"].to_numpy(dtype=float)
    timestamp = pd.to_datetime(df["timestamp"])
    speed_kmh = df["speed_kmh"].to_numpy(dtype=float)

    # Convert back to lat/lon using the SHARED projection -- no origin
    # passed explicitly here since the defaults ARE the agreed IO-VNBD
    # origin. Passing it explicitly anyway for clarity/safety.
    lat, lon = local_xy_to_latlon(x, y, ORIGIN_LAT, ORIGIN_LON)

    duration_s = (timestamp.iloc[-1] - timestamp.iloc[0]).total_seconds()
    path_length_m = np.sqrt(np.diff(x)**2 + np.diff(y)**2).sum()

    print(f"Loaded {len(x)} records from {csv_path}")
    print(f"  duration:    {duration_s/60:.1f} min")
    print(f"  path length: {path_length_m/1000:.2f} km")
    print(f"  x_m range:   [{x.min():.1f}, {x.max():.1f}]")
    print(f"  y_m range:   [{y.min():.1f}, {y.max():.1f}]")
    print(f"  speed range: [{speed_kmh.min():.2f}, {speed_kmh.max():.2f}] km/h")
    print(f"  lat range:   [{np.min(lat):.6f}, {np.max(lat):.6f}]")
    print(f"  lon range:   [{np.min(lon):.6f}, {np.max(lon):.6f}]")

    return {
        "x": x, "y": y, "timestamp": timestamp, "speed_kmh": speed_kmh,
        "lat": lat, "lon": lon,
    }


if __name__ == "__main__":
    import os

    real_file = "S1_training.csv"

    if os.path.exists(real_file):
        print(f"=== Loading REAL IO-VNBD S1 dataset: {real_file} ===")
        data = load_gps_track(real_file)

        print("\nReady to feed data['lat'], data['lon'] into map_matching.py")
        print("(via GraphHopperClient once a real UK-region server is running,")
        print(" or NaiveNearestPointMatcher against a real UK road polyline)")
    else:
        print(f"{real_file} not found -- upload it to run this for real.")

