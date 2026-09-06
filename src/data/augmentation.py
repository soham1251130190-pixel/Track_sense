import numpy as np
from scipy.spatial.transform import Rotation


ACCEL_COLUMNS = [
    "ACCELEROMETER X (m/s²)",
    "ACCELEROMETER Y (m/s²)",
    "ACCELEROMETER Z (m/s²)",
]

GYRO_COLUMNS = [
    "GYROSCOPE Yaw (rad/s)",
    "GYROSCOPE Pitch (rad/s)",
    "GYROSCOPE Roll (rad/s)",
]

GRAVITY_COLUMNS = [
    "GRAVITY X (m/s²)",
    "GRAVITY Y (m/s²)",
    "GRAVITY Z (m/s²)",
]

MAGNETIC_COLUMNS = [
    "MAGNETIC FIELD X (Î¼T)",
    "MAGNETIC FIELD Y (Î¼T)",
    "MAGNETIC FIELD Z (Î¼T)",
]


def add_sensor_augmentation(
    df,
    scale_range=(0.98, 1.02),
    noise_fraction=0.02,
    vibration_amplitude=0.10,
    vibration_frequency_range=(1.0, 3.0),
    orientation_range=(-15.0, 15.0),
    random_seed=42,
):
    """
    Apply realistic smartphone sensor augmentation.

    Augmentations:
        1. Sensor scaling
        2. Gaussian noise
        3. Low-frequency vibration
        4. Consistent random mount orientation

    The original dataframe is never modified.
    """

    rng = np.random.default_rng(random_seed)
    augmented = df.copy()

    # ---------------------------------------------------------
    # 1. SENSOR SCALING + GAUSSIAN NOISE
    # ---------------------------------------------------------

    sensor_columns = ACCEL_COLUMNS + GYRO_COLUMNS

    for column in sensor_columns:

        values = augmented[column].to_numpy(dtype=float)

        scale = rng.uniform(*scale_range)

        noise_std = values.std() * noise_fraction

        noise = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=len(values),
        )

        augmented[column] = values * scale + noise

    # ---------------------------------------------------------
    # 2. VIBRATION OVERLAY
    # ---------------------------------------------------------

    if "TIME SINCE START (ms)" in augmented.columns:

        time_seconds = (
            augmented["TIME SINCE START (ms)"].to_numpy(dtype=float)
            / 1000.0
        )

    else:

        time_seconds = np.arange(len(augmented)) * 0.1

    vibration_frequency = rng.uniform(
        vibration_frequency_range[0],
        vibration_frequency_range[1],
    )

    vibration_phase = rng.uniform(0, 2 * np.pi)

    vibration = vibration_amplitude * np.sin(
        2 * np.pi * vibration_frequency * time_seconds
        + vibration_phase
    )

    for column in ACCEL_COLUMNS:

        axis_factor = rng.uniform(0.7, 1.3)

        augmented[column] += vibration * axis_factor

    # ---------------------------------------------------------
    # 3. RANDOM PHONE MOUNT ORIENTATION
    # ---------------------------------------------------------

    roll = rng.uniform(*orientation_range)
    pitch = rng.uniform(*orientation_range)
    yaw = rng.uniform(*orientation_range)

    rotation = Rotation.from_euler(
        "xyz",
        [roll, pitch, yaw],
        degrees=True,
    )

    # Rotate accelerometer.
    accel = augmented[ACCEL_COLUMNS].to_numpy(dtype=float)
    augmented[ACCEL_COLUMNS] = rotation.apply(accel)

    # Rotate gyroscope.
    gyro = augmented[GYRO_COLUMNS].to_numpy(dtype=float)
    augmented[GYRO_COLUMNS] = rotation.apply(gyro)

    # Rotate gravity vector.
    gravity = augmented[GRAVITY_COLUMNS].to_numpy(dtype=float)
    augmented[GRAVITY_COLUMNS] = rotation.apply(gravity)

    # Rotate magnetic-field vector.
    magnetic = augmented[MAGNETIC_COLUMNS].to_numpy(dtype=float)
    augmented[MAGNETIC_COLUMNS] = rotation.apply(magnetic)

    return augmented


if __name__ == "__main__":

    import sys

    sys.path.insert(
        0,
        str(__file__).replace("\\", "/").rsplit("/src/data", 1)[0]
        + "/src/data"
    )

    from loader import load_io_vnbd

    df = load_io_vnbd()

    augmented_df = add_sensor_augmentation(df)

    print("===== FULL AUGMENTATION TEST =====")
    print(f"Original rows:  {len(df)}")
    print(f"Augmented rows: {len(augmented_df)}")

    # ---------------------------------------------------------
    # Check sensor changes
    # ---------------------------------------------------------

    print("\n===== ORIGINAL vs AUGMENTED =====")

    for column in ACCEL_COLUMNS + GYRO_COLUMNS:

        print(f"\n{column}")
        print(f"  Original std:  {df[column].std():.6f}")
        print(f"  Augmented std: {augmented_df[column].std():.6f}")

    # ---------------------------------------------------------
    # Check gravity magnitude
    # ---------------------------------------------------------

    original_gravity = df[GRAVITY_COLUMNS].to_numpy(dtype=float)
    augmented_gravity = augmented_df[GRAVITY_COLUMNS].to_numpy(dtype=float)

    original_gravity_magnitude = np.linalg.norm(
        original_gravity,
        axis=1,
    )

    augmented_gravity_magnitude = np.linalg.norm(
        augmented_gravity,
        axis=1,
    )

    print("\n===== GRAVITY MAGNITUDE CHECK =====")

    print(
        f"Original mean magnitude:  "
        f"{original_gravity_magnitude.mean():.6f} m/s²"
    )

    print(
        f"Augmented mean magnitude: "
        f"{augmented_gravity_magnitude.mean():.6f} m/s²"
    )

    max_difference = np.max(
        np.abs(
            original_gravity_magnitude
            - augmented_gravity_magnitude
        )
    )

    print(
        f"Maximum magnitude difference: "
        f"{max_difference:.10f} m/s²"
    )