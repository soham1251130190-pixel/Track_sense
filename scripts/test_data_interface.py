import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MODULE = PROJECT_ROOT / "src" / "data"

sys.path.insert(0, str(DATA_MODULE))


from loader import load_io_vnbd
from projection import latlon_to_local_xy
from augmentation import add_sensor_augmentation


# ---------------------------------------------------------
# EXPECTED ML INTERFACE
# ---------------------------------------------------------

EXPECTED_COLUMNS = [
    "timestamp",
    "x_m",
    "y_m",
    "speed_kmh",
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_yaw",
    "gyro_pitch",
    "gyro_roll",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "mag_x",
    "mag_y",
    "mag_z",
]


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

def test_data_interface():

    print("===== DATA INTERFACE TEST =====")

    # Load raw dataset
    df = load_io_vnbd()

    print(f"Raw rows: {len(df)}")

    # GPS -> local XY
    x_m, y_m = latlon_to_local_xy(
        df["GPS LATITUDE (degrees)"].to_numpy(),
        df["GPS LONGITUDE (degrees)"].to_numpy(),
    )

    df["x_m"] = x_m
    df["y_m"] = y_m

    # Apply augmentation
    augmented_df = add_sensor_augmentation(df)

    # Create model-facing dataframe
    model_df = augmented_df[
        [
            "timestamp",
            "x_m",
            "y_m",
            "GPS SPEED (Kmh)",
            "ACCELEROMETER X (m/s²)",
            "ACCELEROMETER Y (m/s²)",
            "ACCELEROMETER Z (m/s²)",
            "GYROSCOPE Yaw (rad/s)",
            "GYROSCOPE Pitch (rad/s)",
            "GYROSCOPE Roll (rad/s)",
            "GRAVITY X (m/s²)",
            "GRAVITY Y (m/s²)",
            "GRAVITY Z (m/s²)",
            "MAGNETIC FIELD X (Î¼T)",
            "MAGNETIC FIELD Y (Î¼T)",
            "MAGNETIC FIELD Z (Î¼T)",
        ]
    ].copy()

    # Rename to ML-friendly names
    model_df = model_df.rename(
        columns={
            "GPS SPEED (Kmh)": "speed_kmh",

            "ACCELEROMETER X (m/s²)": "accel_x",
            "ACCELEROMETER Y (m/s²)": "accel_y",
            "ACCELEROMETER Z (m/s²)": "accel_z",

            "GYROSCOPE Yaw (rad/s)": "gyro_yaw",
            "GYROSCOPE Pitch (rad/s)": "gyro_pitch",
            "GYROSCOPE Roll (rad/s)": "gyro_roll",

            "GRAVITY X (m/s²)": "gravity_x",
            "GRAVITY Y (m/s²)": "gravity_y",
            "GRAVITY Z (m/s²)": "gravity_z",

            "MAGNETIC FIELD X (Î¼T)": "mag_x",
            "MAGNETIC FIELD Y (Î¼T)": "mag_y",
            "MAGNETIC FIELD Z (Î¼T)": "mag_z",
        }
    )

    # ---------------------------------------------------------
    # CHECK 1: COLUMN INTERFACE
    # ---------------------------------------------------------

    print("\n===== COLUMN CHECK =====")

    actual_columns = list(model_df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        print("FAILED ❌")
        print("\nExpected:")
        print(EXPECTED_COLUMNS)

        print("\nActual:")
        print(actual_columns)

        raise AssertionError("Model interface columns do not match.")

    print("Column interface: PASS ✅")

    # ---------------------------------------------------------
    # CHECK 2: ROW COUNT
    # ---------------------------------------------------------

    print("\n===== ROW COUNT CHECK =====")

    if len(model_df) != len(df):
        raise AssertionError("Row count changed during processing.")

    print(f"Rows preserved: {len(model_df)} ✅")

    # ---------------------------------------------------------
    # CHECK 3: MISSING VALUES
    # ---------------------------------------------------------

    print("\n===== MISSING VALUE CHECK =====")

    missing_values = model_df.isna().sum().sum()

    if missing_values != 0:
        raise AssertionError(
            f"Found {missing_values} missing values."
        )

    print("NaN values: 0 ✅")

    # ---------------------------------------------------------
    # CHECK 4: INFINITE VALUES
    # ---------------------------------------------------------

    print("\n===== INFINITE VALUE CHECK =====")

    numeric_data = model_df.select_dtypes(include=np.number)

    infinite_values = np.isinf(numeric_data.to_numpy()).sum()

    if infinite_values != 0:
        raise AssertionError(
            f"Found {infinite_values} infinite values."
        )

    print("Infinite values: 0 ✅")

    # ---------------------------------------------------------
    # CHECK 5: TIMESTAMP ORDER
    # ---------------------------------------------------------

    print("\n===== TIMESTAMP CHECK =====")

    timestamps = pd.to_datetime(model_df["timestamp"])

    if not timestamps.is_monotonic_increasing:
        raise AssertionError("Timestamps are not chronological.")

    print("Timestamps chronological: PASS ✅")

    # ---------------------------------------------------------
    # CHECK 6: SAMPLE MODEL INPUT
    # ---------------------------------------------------------

    print("\n===== SAMPLE MODEL INPUT =====")

    print(model_df.head(3).to_string(index=False))

    print("\n===== RESULT =====")
    print("DATA INTERFACE TEST PASSED ✅")


if __name__ == "__main__":
    test_data_interface()