from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDIAN_FILES = [
    PROJECT_ROOT / "data" / "raw" / "indian.CSV",
    PROJECT_ROOT / "data" / "raw" / "20220806_091545A-02.CSV",
    PROJECT_ROOT / "data" / "raw" / "20220806_091646A-03.CSV",
    PROJECT_ROOT / "data" / "raw" / "20220806_091747A-04.CSV",
    PROJECT_ROOT / "data" / "raw" / "20220806_091848A-05.csv",
]

IOVNBD_FILE = PROJECT_ROOT / "data" / "processed" / "S1_training.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "indian_training.csv"

STANDARD_COLUMNS = [
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


def load_iovnbd_means():
    df = pd.read_csv(IOVNBD_FILE)

    numeric_columns = [
        "x_m",
        "y_m",
        "speed_kmh",
        "gravity_x",
        "gravity_y",
        "gravity_z",
        "mag_x",
        "mag_y",
        "mag_z",
    ]

    return df[numeric_columns].mean()


def load_indian_file(filepath):
    rows = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    for line in lines[1:]:
        line = line.strip()

        if not line or "|G:" not in line:
            continue

        timestamp_text, sensor_text = line.split("|G:", 1)
        values = sensor_text.split(",")

        if len(values) < 7:
            continue

        try:
            values = [float(value) for value in values]
            timestamp_us = int(timestamp_text)

            rows.append(
                {
                    "timestamp": timestamp_us,
                    "accel_x": values[2],
                    "accel_y": values[0],
                    "accel_z": values[1],
                    "gyro_yaw": values[6],
                    "gyro_pitch": values[4],
                    "gyro_roll": values[5],
                }
            )

        except ValueError:
            continue

    return pd.DataFrame(rows)


def load_all_indian_data():
    all_data = []

    for filepath in INDIAN_FILES:
        print(f"Loading: {filepath.name}")

        df = load_indian_file(filepath)

        print(f"  Records: {len(df)}")

        if len(df) == 0:
            raise ValueError(
                f"No valid IMU records found in {filepath.name}"
            )

        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    return combined


def prepare_dataset():
    print("===== LOADING IO-VNBD REFERENCE =====")

    iovnbd_means = load_iovnbd_means()

    print("\nIO-VNBD means used for unavailable fields:")

    for column, value in iovnbd_means.items():
        print(f"{column}: {value:.6f}")

    print("\n===== LOADING INDIAN DriCon DATA =====")

    indian_df = load_all_indian_data()

    print(f"\nTotal Indian IMU records: {len(indian_df)}")

    print("\n===== CREATING STANDARDIZED DATASET =====")

    output_df = pd.DataFrame(index=indian_df.index)

    output_df["timestamp"] = pd.to_datetime(
        indian_df["timestamp"],
        unit="us",
        errors="coerce",
    )

    output_df["x_m"] = iovnbd_means["x_m"]
    output_df["y_m"] = iovnbd_means["y_m"]
    output_df["speed_kmh"] = iovnbd_means["speed_kmh"]

    output_df["accel_x"] = indian_df["accel_x"]
    output_df["accel_y"] = indian_df["accel_y"]
    output_df["accel_z"] = indian_df["accel_z"]

    output_df["gyro_yaw"] = indian_df["gyro_yaw"]
    output_df["gyro_pitch"] = indian_df["gyro_pitch"]
    output_df["gyro_roll"] = indian_df["gyro_roll"]

    output_df["gravity_x"] = iovnbd_means["gravity_x"]
    output_df["gravity_y"] = iovnbd_means["gravity_y"]
    output_df["gravity_z"] = iovnbd_means["gravity_z"]

    output_df["mag_x"] = iovnbd_means["mag_x"]
    output_df["mag_y"] = iovnbd_means["mag_y"]
    output_df["mag_z"] = iovnbd_means["mag_z"]

    output_df = output_df[STANDARD_COLUMNS]

    print("\n===== VALIDATION =====")

    print(f"Rows: {len(output_df)}")
    print(f"Columns: {len(output_df.columns)}")

    missing_columns = [
        column
        for column in STANDARD_COLUMNS
        if column not in output_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing standardized columns: {missing_columns}"
        )

    if output_df[STANDARD_COLUMNS].isnull().any().any():
        raise ValueError("NaN values found in final dataset.")

    numeric_values = output_df.select_dtypes(
        include=[np.number]
    ).to_numpy()

    if np.isinf(numeric_values).any():
        raise ValueError("Infinite values found in final dataset.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n===== DATASET SAVED =====")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    prepare_dataset()