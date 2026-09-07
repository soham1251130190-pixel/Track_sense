from pathlib import Path
import sys

# Add src/data to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MODULE = PROJECT_ROOT / "src" / "data"

sys.path.insert(0, str(DATA_MODULE))

from loader import load_io_vnbd
from projection import latlon_to_local_xy
from augmentation import add_sensor_augmentation
from gnss_quality import calculate_gnss_quality


OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "S1_training.csv"


def prepare_dataset():
    print("===== LOADING DATASET =====")

    df = load_io_vnbd()

    print(f"Loaded rows: {len(df)}")

    # GPS LAT/LON -> LOCAL X/Y
    print("\n===== GPS TO LOCAL XY =====")

    latitude = df["GPS LATITUDE (degrees)"].to_numpy()
    longitude = df["GPS LONGITUDE (degrees)"].to_numpy()

    x_m, y_m = latlon_to_local_xy(latitude, longitude)

    df["x_m"] = x_m
    df["y_m"] = y_m

    print(f"X range: {x_m.min():.2f} to {x_m.max():.2f} m")
    print(f"Y range: {y_m.min():.2f} to {y_m.max():.2f} m")

        # GNSS QUALITY CHECK
    print("\n===== GNSS QUALITY CHECK =====")

    (
        gps_jump_m,
        expected_motion_m,
        motion_ratio,
        gnss_outlier,
    ) = calculate_gnss_quality(
        x=df["x_m"].to_numpy(),
        y=df["y_m"].to_numpy(),
        speed_kmh=df["GPS SPEED (Kmh)"].to_numpy(),
        timestamps=df["timestamp"].values,
    )

    df["gps_jump_m"] = gps_jump_m
    df["expected_motion_m"] = expected_motion_m
    df["motion_ratio"] = motion_ratio
    df["gnss_outlier"] = gnss_outlier

    print(f"GNSS outliers detected: {gnss_outlier.sum()}")
    print(f"Outlier percentage: {gnss_outlier.mean() * 100:.2f}%")

    # SENSOR AUGMENTATION
    print("\n===== SENSOR AUGMENTATION =====")

    augmented_df = add_sensor_augmentation(df)

    print("Augmentation completed.")

    # CREATE CLEAN TRAINING DATASET
    print("\n===== CREATING TRAINING DATASET =====")

    training_df = augmented_df[
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

    # Rename columns to simple ML-friendly names
    training_df = training_df.rename(
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

    # SAVE
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    training_df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved training dataset:")
    print(OUTPUT_FILE)

    print("\n===== FINAL DATASET =====")
    print(f"Rows: {len(training_df)}")
    print(f"Columns: {len(training_df.columns)}")

    print("\nColumns:")
    for column in training_df.columns:
        print(f" - {column}")


if __name__ == "__main__":
    prepare_dataset()