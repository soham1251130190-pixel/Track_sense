import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracksense.eval.datatypes import TrajectoryData, BlackoutInterval
from tracksense.eval.adapters import TCN_EKF_ModelAdapter
from tracksense.eval.baselines import BaselineModels
from tracksense.eval.harness import EvaluationHarness

csv_path = Path("data/S1_training.csv")
if not csv_path.exists():
    csv_path = Path("model2/S1_training.csv")

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path)

# Extract first 3000 samples (300 seconds at 10Hz)
N = min(len(df), 3000)
df = df.iloc[:N]

timestamps = np.arange(len(df)) * 0.1
gt_pos = np.column_stack([df["x_m"].values, df["y_m"].values]).astype(float)
speed_mps = (df["speed_kmh"].values / 3.6).astype(float)

# Compute velocity vector from gt_pos
vx = np.gradient(gt_pos[:, 0], timestamps)
vy = np.gradient(gt_pos[:, 1], timestamps)
gt_vel = np.column_stack([vx, vy])

imu_accel = np.column_stack([df["accel_x"].values, df["accel_y"].values, df["accel_z"].values]).astype(float)
imu_gyro = np.column_stack([df["gyro_yaw"].values, df["gyro_pitch"].values, df["gyro_roll"].values]).astype(float)

# Create 3 blackout intervals: 30s, 60s, 120s
t0 = timestamps[0]
blackouts = [
    BlackoutInterval(start_time=t0 + 20.0, end_time=t0 + 50.0, name="30s_blackout"),
    BlackoutInterval(start_time=t0 + 70.0, end_time=t0 + 130.0, name="60s_blackout"),
    BlackoutInterval(start_time=t0 + 150.0, end_time=t0 + 270.0, name="120s_blackout"),
]

gnss_pos = np.copy(gt_pos)
for b in blackouts:
    mask = (timestamps >= b.start_time) & (timestamps <= b.end_time)
    gnss_pos[mask] = np.nan

gt_data = TrajectoryData(
    timestamps=timestamps,
    gt_position=gt_pos,
    gt_velocity=gt_vel,
    gnss_position=gnss_pos,
    imu_accel=imu_accel,
    imu_gyro=imu_gyro,
    blackout_intervals=blackouts
)

print("\n🚀 Running TrackSense AI+EKF Adapter...")
adapter = TCN_EKF_ModelAdapter(checkpoint_path="model2/best_model_real.pt")
pred_tracksense = adapter.run_pipeline(gt_data)

print("\n🚀 Running Baseline Models...")
pred_naive = BaselineModels.run_naive_dead_reckoning(gt_data)
pred_frozen = BaselineModels.run_frozen_gnss_baseline(gt_data)

harness = EvaluationHarness(target_drift_pct=10.0)
results = harness.run_evaluation(
    gt_data=gt_data,
    predictions=[pred_tracksense],
    output_dir="eval_real_fixed_results",
    include_baselines=True
)

print("\n" + "="*70)
print("EVALUATION RESULTS AFTER FIX:")
print("="*70)
for name, (metrics, status) in results.items():
    print(f"\nModel: {name}")
    print(f"  Overall RMSE:        {metrics.overall_rmse_m:.2f} m")
    print(f"  Overall MAE:         {metrics.overall_mae_m:.2f} m")
    print(f"  Mean Blackout Drift: {metrics.mean_drift_percentage:.2f}%")
    print(f"  Max Blackout Drift:  {metrics.max_drift_percentage:.2f}%")
    for sm in metrics.segment_metrics:
        print(f"    Segment {sm.interval.name} ({sm.blackout_duration_s:.1f}s, dist={sm.distance_traveled_m:.1f}m): Error={sm.end_drift_distance_m:.2f}m, Drift={sm.drift_percentage:.2f}% (Pass: {sm.passed_target})")

