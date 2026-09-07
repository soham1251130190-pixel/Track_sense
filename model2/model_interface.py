"""
model_interface.py
Defines the exact data interface between Member 1's pipeline and Person 2's TCN model.

This file serves as a CONTRACT between the data pipeline and the ML model.
No assumptions - all inputs and outputs are explicitly specified.

INPUT FORMAT (from Member 1):
    CSV with at least these columns:
        - accel_x (m/s^2)
        - accel_y (m/s^2)
        - accel_z (m/s^2)
        - gyro_yaw (rad/s)      # Rotation around Z axis
        - gyro_pitch (rad/s)    # Rotation around Y axis
        - gyro_roll (rad/s)     # Rotation around X axis
        - speed_kmh (km/h)      # Ground truth speed for training

OUTPUT FORMAT (from Person 2's model):
    (batch_size, 3) where:
        - index 0: speed_mean (km/h) - predicted forward velocity
        - index 1: speed_log_var - uncertainty estimate (log variance)
        - index 2: heading_rate (rad/s) - rate of heading change

MODEL INPUT SHAPE:
    (batch_size, 6, window_size)
    where window_size = 100 (10 seconds at 10Hz)

OPTIONAL (for EKF, NOT used by TCN):
    - x_m, y_m (position)
    - mag_x, mag_y, mag_z (magnetometer)
    - gravity_x, gravity_y, gravity_z
    - timestamp
"""

# Required columns for model input
MODEL_INPUT_COLUMNS = [
    'accel_x',      # m/s^2
    'accel_y',      # m/s^2
    'accel_z',      # m/s^2
    'gyro_yaw',     # rad/s
    'gyro_pitch',   # rad/s
    'gyro_roll',    # rad/s
]

# Target column for training
MODEL_TARGET = 'speed_kmh'  # km/h (converted to m/s internally)

# Optional columns used by EKF (NOT used by TCN)
EKF_EXTRA_COLUMNS = [
    'x_m',          # Local X position (m)
    'y_m',          # Local Y position (m)
    'mag_x',        # Magnetometer X (uT)
    'mag_y',        # Magnetometer Y (uT)
    'mag_z',        # Magnetometer Z (uT)
    'gravity_x',    # Gravity vector X (m/s^2)
    'gravity_y',    # Gravity vector Y (m/s^2)
    'gravity_z',    # Gravity vector Z (m/s^2)
    'timestamp',    # Time index
]

# Model input shape
MODEL_INPUT_SHAPE = (None, 6, 100)  # (batch_size, features, window_size)

# Model output shape
MODEL_OUTPUT_SHAPE = (None, 3)  # (batch_size, [speed_mean, log_var, heading_rate])


def validate_interface_columns(df):
    """
    Validate that the DataFrame contains all required columns.

    Args:
        df: Pandas DataFrame from Member 1's pipeline

    Returns:
        bool: True if all required columns are present

    Raises:
        ValueError: If required columns are missing
    """
    missing = [col for col in MODEL_INPUT_COLUMNS + [MODEL_TARGET]
               if col not in df.columns]

    if missing:
        print(f"[FAIL] Missing required columns: {missing}")
        print(f"   Available columns: {df.columns.tolist()}")
        print(f"   Expected: {MODEL_INPUT_COLUMNS + [MODEL_TARGET]}")
        return False

    print(f"[OK] All required columns present")
    print(f"   {len(MODEL_INPUT_COLUMNS)} features + 1 target")
    return True


def prepare_model_input(df, window_size=100):
    """
    Prepare input for the TCN model from a DataFrame.

    This is the official interface between Member 1's pipeline
    and Person 2's model.

    Args:
        df: Pandas DataFrame with required columns
        window_size: Number of timesteps per window (100 = 10s)

    Returns:
        X: (n_samples, 6, window_size) - Input features
        y: (n_samples,) - Target speeds in km/h

    Raises:
        ValueError: If columns are missing or data is invalid
    """
    import numpy as np

    # Validate columns
    if not validate_interface_columns(df):
        raise ValueError("Invalid column format")

    # Check for NaNs/Infs
    selected_cols = MODEL_INPUT_COLUMNS + [MODEL_TARGET]
    if df[selected_cols].isna().any().any():
        raise ValueError("Data contains NaN values")
    if np.isinf(df[selected_cols].values).any():
        raise ValueError("Data contains Inf values")

    # Use speed directly in km/h (no conversion)
    speed_kmh = df[MODEL_TARGET].values

    X = []
    y = []

    for i in range(window_size, len(df)):
        window = df[MODEL_INPUT_COLUMNS].iloc[i-window_size:i].values
        X.append(window.T)  # (6, window_size)
        y.append(speed_kmh[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"[OK] Prepared model input")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Speed range: {y.min():.2f} - {y.max():.2f} km/h")

    return X, y


# Test the interface
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    print("=" * 60)
    print("INTERFACE VALIDATION TEST")
    print("=" * 60)

    # Create test DataFrame with required columns
    n_samples = 500
    test_df = pd.DataFrame({
        'accel_x': np.random.randn(n_samples),
        'accel_y': np.random.randn(n_samples),
        'accel_z': np.random.randn(n_samples) + 9.8,
        'gyro_yaw': np.random.randn(n_samples) * 0.05,
        'gyro_pitch': np.random.randn(n_samples) * 0.03,
        'gyro_roll': np.random.randn(n_samples) * 0.03,
        'speed_kmh': np.random.uniform(20, 80, n_samples),
        'x_m': np.cumsum(np.random.randn(n_samples) * 10),
        'y_m': np.cumsum(np.random.randn(n_samples) * 10),
        'mag_x': np.random.randn(n_samples) * 0.5,
        'mag_y': np.random.randn(n_samples) * 0.5,
        'mag_z': np.random.randn(n_samples) * 0.5,
        'timestamp': np.arange(n_samples),
    })

    print(f"[OK] Test DataFrame created with {len(test_df)} rows")

    # Test interface
    try:
        X, y = prepare_model_input(test_df, window_size=50)
        print(f"\n[OK] Interface test passed!")
        print(f"   Model expects: {MODEL_INPUT_SHAPE}")
        print(f"   Actual X shape: {X.shape}")
        print(f"   Target y shape: {y.shape}")
        print(f"   Target unit: km/h")
    except Exception as e:
        print(f"\n[FAIL] Interface test failed: {e}")
        
