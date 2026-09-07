"""
evaluate_real.py
Comprehensive evaluation of trained TCN on real IO-VNBD test data with EKF fusion.

Evaluates:
1. Pure IMU Open-Loop Dead Reckoning (No TCN updates)
2. TCN-Aided EKF Trajectory (TCN velocity updates in km/h -> m/s)
3. Comparison with Ground Truth (x_m, y_m)
4. Metrics: Position Drift (m), Trajectory RMSE (m), Speed RMSE (km/h)
5. Plots: real_ekf_evaluation.png & real_velocity_comparison.png
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import build_model
from dummy_ekf import DummyEKF
from real_data_loader import load_real_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_real_evaluation(checkpoint_path=None, csv_path=None, max_eval_steps=1000):
    """
    Run evaluation on real hold-out test sequence.
    """
    print("=" * 60)
    print("REAL DATA EVALUATION WITH EKF (HOURS 6-10)")
    print("=" * 60)
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(SCRIPT_DIR, 'best_model_real.pt')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(SCRIPT_DIR, 'best_model_synthetic.pt')
            
    print(f"📦 Loading Model Checkpoint from: {checkpoint_path}")
    model = build_model()
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 1. Load real test set
    _, _, test_loader, meta = load_real_data(
        csv_path=csv_path,
        window_size=100,
        val_split=0.15,
        test_split=0.15,
        batch_size=256
    )
    
    X_test = meta['X_test']
    y_test = meta['y_test']
    extra_test = meta['extra_test']
    
    n_steps = min(len(X_test), max_eval_steps)
    print(f"\n📊 Evaluating on {n_steps} sequential test timesteps...")
    
    X_seq = X_test[:n_steps]
    y_seq = y_test[:n_steps]  # Speed in km/h
    
    # 2. Run TCN Inference
    X_tensor = torch.tensor(X_seq, dtype=torch.float32)
    with torch.no_grad():
        output = model(X_tensor)
        
    speed_mean_kmh = output[:, 0].numpy()
    speed_log_var = output[:, 1].numpy()
    heading_rate = output[:, 2].numpy()
    
    # Speed Metrics
    speed_rmse_kmh = np.sqrt(np.mean((speed_mean_kmh - y_seq) ** 2))
    speed_mae_kmh = np.mean(np.abs(speed_mean_kmh - y_seq))
    speed_rmse_mps = speed_rmse_kmh / 3.6
    speed_mae_mps = speed_mae_kmh / 3.6
    
    print(f"\n⚡ TCN Speed Prediction Metrics on Real Test Set:")
    print(f"   Speed RMSE: {speed_rmse_kmh:.2f} km/h ({speed_rmse_mps:.2f} m/s)")
    print(f"   Speed MAE:  {speed_mae_kmh:.2f} km/h ({speed_mae_mps:.2f} m/s)")
    
    # 3. Prepare IMU Data for EKF
    # X_seq shape: (n_steps, 6, 100) -> last timestep features:
    # 0: accel_x, 1: accel_y, 2: accel_z, 3: gyro_yaw, 4: gyro_pitch, 5: gyro_roll
    imu_data = []
    for i in range(n_steps):
        ax = float(X_seq[i, 0, -1])
        ay = float(X_seq[i, 1, -1])
        gz = float(X_seq[i, 3, -1])  # gyro_yaw
        imu_data.append((ax, ay, gz))
        
    # 4. Prepare TCN Velocity Updates (km/h -> m/s)
    velocity_predictions = []
    uncertainties = []
    ekf_heading = 0.0
    dt = 0.1
    
    for i in range(n_steps):
        speed_kmh = speed_mean_kmh[i]
        speed_mps = speed_kmh / 3.6  # Convert to m/s for EKF
        gz = imu_data[i][2]
        ekf_heading += gz * dt
        
        vx = speed_mps * np.cos(ekf_heading)
        vy = speed_mps * np.sin(ekf_heading)
        velocity_predictions.append((vx, vy))
        
        var_kmh2 = np.exp(speed_log_var[i])
        var_mps2 = max(var_kmh2 / (3.6 ** 2), 1e-3)
        uncertainties.append(var_mps2)
        
    # 5. Run Open-Loop EKF (Pure IMU dead reckoning)
    print("\n🚀 Running Open-Loop EKF (No Velocity Updates)...")
    ekf_open = DummyEKF(dt=dt)
    traj_open = ekf_open.run_open_loop(imu_data)
    final_pos_open = ekf_open.get_position()
    
    # 6. Run EKF with TCN Velocity Updates
    print("🚀 Running TCN-Aided EKF (With TCN Velocity Updates)...")
    ekf_tcn = DummyEKF(dt=dt)
    traj_tcn = ekf_tcn.run_with_updates(imu_data, velocity_predictions, uncertainties)
    final_pos_tcn = ekf_tcn.get_position()
    
    # 7. Ground Truth Relative Trajectory (if available)
    traj_gt = None
    if extra_test['x_m'] is not None and extra_test['y_m'] is not None:
        gt_x = extra_test['x_m'][:n_steps]
        gt_y = extra_test['y_m'][:n_steps]
        # Normalize relative to start (0, 0)
        gt_x_rel = gt_x - gt_x[0]
        gt_y_rel = gt_y - gt_y[0]
        traj_gt = list(zip(gt_x_rel, gt_y_rel))
        
    # 8. Compare Drift & Performance
    drift_open = np.linalg.norm(np.array(final_pos_open))
    drift_tcn = np.linalg.norm(np.array(final_pos_tcn))
    
    traj_open_arr = np.array(traj_open)
    traj_tcn_arr = np.array(traj_tcn)
    
    print("\n" + "=" * 60)
    print("EKF INTEGRATION RESULTS ON REAL DATA")
    print("=" * 60)
    print(f"Final Open-Loop Drift:    {drift_open:.2f} m")
    print(f"Final TCN-Aided Drift:    {drift_tcn:.2f} m")
    
    if traj_gt is not None:
        gt_arr = np.array(traj_gt)
        error_open = np.linalg.norm(traj_open_arr[-1] - gt_arr[-1])
        error_tcn = np.linalg.norm(traj_tcn_arr[-1] - gt_arr[-1])
        reduction = (1.0 - error_tcn / max(error_open, 1e-4)) * 100
        print(f"Ground Truth Final Error (Open Loop): {error_open:.2f} m")
        print(f"Ground Truth Final Error (With TCN):  {error_tcn:.2f} m")
        print(f"🌟 Trajectory Error Reduction:        {reduction:.1f}%")
        
    # 9. Plotting Trajectory Comparison
    plot_trajectory(traj_open_arr, traj_tcn_arr, traj_gt, n_steps)
    
    # 10. Plotting Speed Comparison
    plot_speed_comparison(y_seq, speed_mean_kmh, speed_log_var, n_steps)
    
    return {
        'speed_rmse_kmh': float(speed_rmse_kmh),
        'speed_mae_kmh': float(speed_mae_kmh),
        'speed_rmse_mps': float(speed_rmse_mps),
        'drift_open': float(drift_open),
        'drift_tcn': float(drift_tcn),
    }


def plot_trajectory(traj_open, traj_tcn, traj_gt, n_steps):
    """Plot trajectory comparison."""
    plt.figure(figsize=(10, 8))
    
    plt.plot(traj_open[:, 0], traj_open[:, 1], 'r--', label='Pure IMU (Open Loop)', linewidth=2, alpha=0.7)
    plt.plot(traj_tcn[:, 0], traj_tcn[:, 1], 'g-', label='TCN + EKF (Proposed)', linewidth=2.5)
    
    if traj_gt is not None:
        gt_arr = np.array(traj_gt)
        plt.plot(gt_arr[:, 0], gt_arr[:, 1], 'k:', label='Ground Truth Reference', linewidth=2)
        
    plt.scatter([0], [0], c='black', marker='o', s=120, label='Start Position')
    plt.scatter(traj_open[-1, 0], traj_open[-1, 1], c='red', marker='s', s=100, label='End (Open Loop)')
    plt.scatter(traj_tcn[-1, 0], traj_tcn[-1, 1], c='green', marker='s', s=100, label='End (TCN+EKF)')
    
    plt.xlabel('Local X Position (m)')
    plt.ylabel('Local Y Position (m)')
    plt.title(f'Real IO-VNBD Trajectory Comparison ({n_steps} timesteps @ 10Hz)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    
    save_path = os.path.join(SCRIPT_DIR, 'real_ekf_evaluation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Trajectory comparison plot saved to: {save_path}")


def plot_speed_comparison(y_true, y_pred, log_var, n_steps):
    """Plot predicted speed vs ground truth speed with uncertainty band."""
    plt.figure(figsize=(12, 5))
    
    time_axis = np.arange(n_steps) * 0.1  # seconds
    std = np.sqrt(np.exp(log_var))
    
    plt.plot(time_axis, y_true, 'k-', label='Ground Truth Speed (km/h)', linewidth=1.8)
    plt.plot(time_axis, y_pred, 'b-', label='TCN Predicted Speed (km/h)', linewidth=1.8, alpha=0.85)
    plt.fill_between(time_axis, y_pred - std, y_pred + std, color='b', alpha=0.2, label='1σ Uncertainty Band')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Forward Speed (km/h)')
    plt.title('Real-Time Speed Estimation vs Ground Truth (IO-VNBD Dataset)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(SCRIPT_DIR, 'real_velocity_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Speed estimation plot saved to: {save_path}")


if __name__ == "__main__":
    run_real_evaluation()

