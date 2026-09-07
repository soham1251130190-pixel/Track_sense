# Person 2 - ML Model (TCN for Velocity Estimation)

## Overview

Temporal Convolutional Network (TCN) for IMU-based velocity estimation in GNSS-denied environments.

**Architecture:** 5 TCN residual blocks → Global Average Pooling → MLP Head  
**Input:** `(batch, 6, 100)` — 6 IMU channels × 100 timesteps (10s @ 10Hz)  
**Output:** `(batch, 3)` — `[speed_mean, speed_log_var, heading_rate]`  
**Parameters:** ~100K (target)

---

## Quick Setup (5 minutes)

### 1. Create virtual environment

```bash
python -m venv venv_tcn
source venv_tcn/bin/activate    # Linux/Mac
# OR
venv_tcn\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

### 3. Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} ✅')"
```

---

## Files in This Package

| File                   | Purpose                        |
|------------------------|--------------------------------|
| `model.py`             | TCN model architecture         |
| `test_synthetic.py`    | Quick tests on synthetic data  |
| `synthetic_training.py`| Training loop with fake data   |
| `model_interface.py`   | Input/output specification     |
| `README.md`            | This file                      |

---

## Quick Test (2 minutes)

```bash
cd model
python model.py               # Self-test: instantiation + forward pass
python test_synthetic.py       # Full test suite (4 tests)
```

**Expected output:** `🎉 All tests passed! Model is ready for real data.`

---

## Synthetic Training (5 minutes)

```bash
python synthetic_training.py
```

**Expected output:** Training curves saved to `training_curves.png` + best model saved to `best_model_synthetic.pt`.

---

## Interface with Member 1's Data

Member 1's CSV must contain these columns:

| Column       | Unit   | Description                    |
|-------------|--------|--------------------------------|
| `accel_x`   | m/s²   | Forward acceleration           |
| `accel_y`   | m/s²   | Lateral acceleration           |
| `accel_z`   | m/s²   | Vertical acceleration          |
| `gyro_yaw`  | rad/s  | Rotation around Z axis         |
| `gyro_pitch` | rad/s | Rotation around Y axis         |
| `gyro_roll` | rad/s  | Rotation around X axis         |
| `speed_kmh` | km/h   | Ground truth speed (training)  |

### Usage

```python
from model_interface import prepare_model_input
from model import build_model
import pandas as pd
import torch

# Load data from Member 1's pipeline
df = pd.read_csv("imu_data.csv")

# Prepare input
X, y = prepare_model_input(df, window_size=100)

# Run inference
model = build_model()
model.load_state_dict(torch.load("best_model_synthetic.pt"))
model.eval()

with torch.no_grad():
    output = model(torch.tensor(X))
    speed_mean = output[:, 0]      # km/h
    speed_log_var = output[:, 1]   # uncertainty
    heading_rate = output[:, 2]    # rad/s
```

---

## Model Output Specification

| Index | Name            | Unit   | Description                          |
|-------|-----------------|--------|--------------------------------------|
| 0     | `speed_mean`    | km/h   | Predicted forward velocity           |
| 1     | `speed_log_var` | —      | Log variance (uncertainty for EKF)   |
| 2     | `heading_rate`  | rad/s  | Rate of heading change               |

---

## Troubleshooting

| Problem              | Solution                                    |
|---------------------|---------------------------------------------|
| "Module not found"  | Run `pip install torch numpy pandas matplotlib` |
| "CUDA not available"| Fine — will use CPU automatically           |
| "Memory error"      | Reduce `BATCH_SIZE` in `synthetic_training.py` |
| Tests fail          | Check PyTorch version ≥ 1.9                 |

