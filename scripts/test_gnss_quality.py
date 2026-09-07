import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.loader import load_io_vnbd
from data.projection import latlon_to_local_xy
from data.gnss_quality import calculate_gnss_quality


data = load_io_vnbd()

x, y = latlon_to_local_xy(
    data["GPS LATITUDE (degrees)"],
    data["GPS LONGITUDE (degrees)"],
)

(
    gps_jump_m,
    expected_motion_m,
    motion_ratio,
    gnss_outlier,
) = calculate_gnss_quality(
    x=x,
    y=y,
    speed_kmh=data["GPS SPEED (Kmh)"],
    timestamps=data["timestamp"].values,
)

print("===== GNSS QUALITY TEST =====")
print(f"Total samples: {len(data)}")
print(f"GNSS outliers detected: {gnss_outlier.sum()}")
print(f"Outlier percentage: {gnss_outlier.mean() * 100:.2f}%")

print("\n===== MOTION RATIO =====")
valid_ratio = motion_ratio[~pd.isna(motion_ratio)]
print(pd.Series(valid_ratio).describe())

print("\n===== RESULT =====")

if gnss_outlier.sum() > 0:
    print("GNSS QUALITY DETECTOR WORKING ✅")
else:
    print("GNSS QUALITY DETECTOR FAILED ❌")