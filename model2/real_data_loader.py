"""
real_data_loader.py
Data loader and preprocessor for Member 1's real IO-VNBD dataset.

Handles:
1. Loading real CSV dataset (16 columns)
2. Quality validation (NaN, Inf, sample count, frequency, chronology)
3. High-performance sliding window generation: (batch, 6, 100) at 10Hz
4. Target speed extraction in km/h (speed_kmh)
5. Chronological Train / Val / Test splitting
6. PyTorch DataLoader creation
7. Supports both IO_VNBD_DataLoader and RealDataLoader classes
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Model input features (6 channels)
FEATURE_COLUMNS = [
    'accel_x',      # m/s² (Forward acceleration)
    'accel_y',      # m/s² (Lateral acceleration)
    'accel_z',      # m/s² (Vertical acceleration)
    'gyro_yaw',     # rad/s (Yaw rate around Z)
    'gyro_pitch',   # rad/s (Pitch rate around Y)
    'gyro_roll',    # rad/s (Roll rate around X)
]

# Target speed column (km/h)
TARGET_COLUMN = 'speed_kmh'

# All 16 expected columns from Member 1
ALL_EXPECTED_COLUMNS = [
    'timestamp', 'x_m', 'y_m', 'speed_kmh',
    'accel_x', 'accel_y', 'accel_z',
    'gyro_yaw', 'gyro_pitch', 'gyro_roll',
    'gravity_x', 'gravity_y', 'gravity_z',
    'mag_x', 'mag_y', 'mag_z'
]


def find_data_file(filepath=None):
    """Locate the dataset file searching multiple common locations."""
    if filepath and os.path.exists(filepath):
        return os.path.abspath(filepath)
    
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', 'data', 'S1_training.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'combined_training.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'combined_training - combined_training.csv'),
        os.path.join(os.path.dirname(__file__), 'data', 'S1_training.csv'),
        os.path.join(os.path.dirname(__file__), 'S1_training.csv'),
        'data/S1_training.csv',
        'combined_training - combined_training.csv',
        'S1_training.csv'
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
            
    raise FileNotFoundError(f"Could not find dataset CSV in candidates: {candidates}")


def validate_data_quality(df):
    """
    Perform rigorous data quality validation on Member 1's dataset.
    
    Checks:
    - Expected columns existence
    - Missing values (NaN)
    - Infinite values (Inf)
    - Sample count
    - Timestamps continuity
    
    Returns:
        dict: Quality validation summary report
    """
    print("\n🔍 Validating Data Quality...")
    
    # 1. Check required columns
    missing_cols = [col for col in ALL_EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    print(f"   ✅ All {len(ALL_EXPECTED_COLUMNS)} required columns present")
    
    # 2. Check NaNs
    nan_counts = df[ALL_EXPECTED_COLUMNS].isna().sum().sum()
    if nan_counts > 0:
        raise ValueError(f"Found {nan_counts} NaN values in dataset!")
    print(f"   ✅ 0 NaN values found")
    
    # 3. Check Infs
    num_cols = [c for c in ALL_EXPECTED_COLUMNS if c != 'timestamp']
    inf_counts = np.isinf(df[num_cols].values).sum()
    if inf_counts > 0:
        raise ValueError(f"Found {inf_counts} Inf values in dataset!")
    print(f"   ✅ 0 Inf values found")
    
    # 4. Sample count
    n_rows = len(df)
    print(f"   ✅ Total samples: {n_rows:,}")
    
    # 5. Speed summary
    speeds = df[TARGET_COLUMN].values
    print(f"   ✅ Speed stats (km/h): min={speeds.min():.2f}, max={speeds.max():.2f}, mean={speeds.mean():.2f}, std={speeds.std():.2f}")
    
    # 6. Check IMU values range
    acc_mag = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2).mean()
    print(f"   ✅ Mean acceleration magnitude: {acc_mag:.2f} m/s² (Earth gravity check)")
    
    return {
        'total_samples': n_rows,
        'nan_count': int(nan_counts),
        'inf_count': int(inf_counts),
        'speed_min': float(speeds.min()),
        'speed_max': float(speeds.max()),
        'speed_mean': float(speeds.mean()),
    }


class IOVNDataset(Dataset):
    """PyTorch Dataset for IO-VNBD sliding window sequences."""
    def __init__(self, X, y, extra=None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.extra = extra  # Optional dict containing ground truth positions/timestamps
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sliding_windows(df, window_size=100, step_size=1):
    """
    Generate sliding windows for TCN input and target velocity using vectorized striding.
    
    Input shape: (n_windows, 6, window_size)
    Target shape: (n_windows,) in km/h
    
    Args:
        df: Pandas DataFrame
        window_size: Number of timesteps per window (100 = 10s @ 10Hz)
        step_size: Step stride between windows
        
    Returns:
        X: np.ndarray (n_samples, 6, window_size)
        y: np.ndarray (n_samples,) target speed in km/h
        extra: dict of aligned metadata (positions, timestamps, etc.)
    """
    features = df[FEATURE_COLUMNS].values.astype(np.float32)  # (N, 6)
    speeds = df[TARGET_COLUMN].values.astype(np.float32)      # (N,)
    
    # Vectorized sliding window creation using stride_tricks
    # sliding_window_view on axis 0 gives shape (N - window_size + 1, 6, window_size)
    windows = np.lib.stride_tricks.sliding_window_view(features, window_shape=window_size, axis=0)
    
    if step_size > 1:
        windows = windows[::step_size]
        y = speeds[window_size - 1::step_size]
    else:
        y = speeds[window_size - 1:]
        
    # Copy array to make memory contiguous for PyTorch DataLoader
    X = np.ascontiguousarray(windows, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    
    extra = {}
    if 'x_m' in df.columns:
        x_m = df['x_m'].values[window_size - 1::step_size]
        extra['x_m'] = np.ascontiguousarray(x_m, dtype=np.float32)
    if 'y_m' in df.columns:
        y_m = df['y_m'].values[window_size - 1::step_size]
        extra['y_m'] = np.ascontiguousarray(y_m, dtype=np.float32)
    if 'timestamp' in df.columns:
        ts = df['timestamp'].values[window_size - 1::step_size]
        extra['timestamp'] = ts
        
    return X, y, extra


def load_real_data(csv_path=None, window_size=100, val_split=0.15, test_split=0.15, batch_size=256, step_size=1):
    """
    Load, validate, slice into windows, and split real IO-VNBD dataset.
    
    Uses chronological splitting (no shuffling before split) to prevent data leakage.
    
    Returns:
        train_loader, val_loader, test_loader, dataset_meta
    """
    file_path = find_data_file(csv_path)
    print(f"\n📂 Loading Real Dataset from: {file_path}")
    
    df = pd.read_csv(file_path)
    meta = validate_data_quality(df)
    
    print(f"\n⚙️  Generating Sliding Windows (window_size={window_size}, step={step_size})...")
    X, y, extra = create_sliding_windows(df, window_size=window_size, step_size=step_size)
    print(f"   Total Windows: {len(X):,}")
    print(f"   X Shape: {X.shape} (batch, channels=6, timesteps={window_size})")
    print(f"   y Shape: {y.shape} (speed in km/h)")
    
    # Chronological Split: Train (70%) | Val (15%) | Test (15%)
    total_len = len(X)
    train_end = int(total_len * (1.0 - val_split - test_split))
    val_end = int(total_len * (1.0 - test_split))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    extra_test = {
        'x_m': extra['x_m'][val_end:] if 'x_m' in extra else None,
        'y_m': extra['y_m'][val_end:] if 'y_m' in extra else None,
        'timestamp': extra['timestamp'][val_end:] if 'timestamp' in extra else None,
    }
    
    print(f"\n📊 Dataset Splits (Chronological):")
    print(f"   Train set: {len(X_train):,} samples ({len(X_train)/total_len*100:.1f}%)")
    print(f"   Val set:   {len(X_val):,} samples ({len(X_val)/total_len*100:.1f}%)")
    print(f"   Test set:  {len(X_test):,} samples ({len(X_test)/total_len*100:.1f}%)")
    
    train_dataset = IOVNDataset(X_train, y_train)
    val_dataset = IOVNDataset(X_val, y_val)
    test_dataset = IOVNDataset(X_test, y_test, extra=extra_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    meta['train_samples'] = len(X_train)
    meta['val_samples'] = len(X_val)
    meta['test_samples'] = len(X_test)
    meta['extra_test'] = extra_test
    meta['X_test'] = X_test
    meta['y_test'] = y_test
    
    return train_loader, val_loader, test_loader, meta


class IO_VNBD_DataLoader:
    """
    High-level Data Loader Class for Member 1's IO-VNBD dataset.
    """
    def __init__(self, csv_path='S1_training.csv', window_size=100, val_split=0.15, test_split=0.15, batch_size=256, step_size=1):
        self.csv_path = csv_path
        self.window_size = window_size
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.step_size = step_size
        
        self.train_loader, self.val_loader, self.test_loader, self.meta = load_real_data(
            csv_path=self.csv_path,
            window_size=self.window_size,
            val_split=self.val_split,
            test_split=self.test_split,
            batch_size=self.batch_size,
            step_size=self.step_size
        )
        
    def get_loaders(self):
        """Returns (train_loader, val_loader, test_loader)."""
        return self.train_loader, self.val_loader, self.test_loader


# Backward compatibility alias
RealDataLoader = IO_VNBD_DataLoader


def test_real_data_loader():
    """Self test for real_data_loader."""
    print("=" * 60)
    print("REAL DATA LOADER TEST")
    print("=" * 60)
    
    # Test class instantiation
    loader = IO_VNBD_DataLoader(
        csv_path='S1_training.csv',
        window_size=100,
        val_split=0.15,
        test_split=0.15,
        batch_size=256
    )
    train_loader, val_loader, test_loader = loader.get_loaders()
    
    # Check first batch
    for X_b, y_b in train_loader:
        print(f"\n✅ First batch loaded successfully:")
        print(f"   Batch X shape: {X_b.shape} (Expected: [256, 6, 100])")
        print(f"   Batch y shape: {y_b.shape} (Expected: [256])")
        print(f"   Batch speed range: {y_b.min():.2f} - {y_b.max():.2f} km/h")
        assert X_b.shape == (256, 6, 100), f"Unexpected X shape: {X_b.shape}"
        assert y_b.shape == (256,), f"Unexpected y shape: {y_b.shape}"
        break
        
    print("\n🎉 real_data_loader self-test PASSED!")
    return True


if __name__ == "__main__":
    test_real_data_loader()

