"""
run_tuned_ekf.py
----------------
Run the final tuned EKF on real S1 data and report performance.
"""

import numpy as np
import pandas as pd
from ekf_fusion_tuned import VehicleEKFTuned
from p2_model_integration import P2ModelWrapper
import matplotlib.pyplot as plt

def run_tuned_ekf(csv_path="S1_training.csv", model_path="model/tcn_velocity_model.onnx"):
    """Run the final tuned EKF and report performance."""
    
    print("\n" + "="*60)
    print("FINAL TUNED EKF - PERFORMANCE EVALUATION")
    print("="*60)
    
    # Load data
    df = pd.read_csv(csv_path)
    timestamps = pd.to_datetime(df["timestamp"])
    dt_array = timestamps.diff().dt.total_seconds().fillna(0.1).to_numpy()
    
    x_m = df["x_m"].to_numpy(dtype=float)
    y_m = df["y_m"].to_numpy(dtype=float)
    speed_kmh = df["speed_kmh"].to_numpy(dtype=float)
    gyro_yaw = df["gyro_yaw"].to_numpy(dtype=float)
    
    # GPS update mask
    gps_update_mask = np.zeros(len(df), dtype=bool)
    gps_update_mask[0] = True
    gps_update_mask[1:] = (np.diff(x_m) != 0) | (np.diff(y_m) != 0)
    real_gps_indices = np.where(gps_update_mask)[0]
    
    print(f"\nData: {len(df)} rows, {len(real_gps_indices)} real GPS fixes")
    
    # Initial state
    x0, y0 = x_m[real_gps_indices[0]], y_m[real_gps_indices[0]]
    x1, y1 = x_m[real_gps_indices[1]], y_m[real_gps_indices[1]]
    init_heading = np.arctan2(y1 - y0, x1 - x0)
    
    # Initialize tuned EKF
    ekf = VehicleEKFTuned(initial_state=[x0, y0, speed_kmh[0] / 3.6, init_heading])
    model_wrapper = P2ModelWrapper(model_path, use_onnx=True)
    
    # Run
    estimated_heading = init_heading
    estimated_positions = []
    pre_errors = []
    nis_values = []
    
    for i in range(len(df)):
        dt = max(dt_array[i], 1e-3)
        ekf.predict(dt)
        
        # Model prediction
        imu_window = model_wrapper.build_window(df, i)
        if imu_window is not None:
            speed_mps, heading_rate, _, log_var = model_wrapper.get_ekf_update(imu_window)
            estimated_heading += heading_rate * dt
            ekf.update_velocity(speed_mps * 3.6, estimated_heading % (2 * np.pi), log_var)
        else:
            estimated_heading += gyro_yaw[i] * dt
            ekf.update_velocity(speed_kmh[i], estimated_heading % (2 * np.pi))
        
        # GPS update
        if gps_update_mask[i] and i > 0:
            # Capture pre-correction error
            pre_err = np.linalg.norm(ekf.x[:2] - np.array([x_m[i], y_m[i]]))
            pre_errors.append(pre_err)
            
            # Compute NIS for this update
            z = np.array([x_m[i], y_m[i]])
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            y = z - H @ ekf.x
            S = H @ ekf.P @ H.T + ekf.R_gps
            try:
                nis = float(y.T @ np.linalg.inv(S) @ y)
                nis_values.append(nis)
            except:
                nis_values.append(float('inf'))
            
            ekf.update_gps(x_m[i], y_m[i])
        
        estimated_positions.append(ekf.x[:2].copy())
    
    estimated_positions = np.array(estimated_positions)
    pre_errors = np.array(pre_errors)
    nis_values = np.array(nis_values)
    
    # Report results
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    
    print(f"\n[NIS DIAGNOSTIC]")
    print(f"  Mean NIS: {nis_values.mean():.3f}  (target: 2.0)")
    print(f"  Median NIS: {np.median(nis_values):.3f}  (target: 1.39)")
    print(f"  % > 5.99: {100 * sum(nis_values > 5.99) / len(nis_values):.1f}%  (target: 5%)")
    
    print(f"\n[PRE-CORRECTION ERROR] - Dead-reckoning drift per GPS gap")
    print(f"  Mean:   {pre_errors.mean():.2f} m")
    print(f"  Median: {np.median(pre_errors):.2f} m")
    print(f"  Std:    {pre_errors.std():.2f} m")
    print(f"  Max:    {pre_errors.max():.2f} m")
    
    gps_pct = 100 * ekf.n_gps_rejected / ekf.n_gps_updates if ekf.n_gps_updates else 0
    vel_pct = 100 * ekf.n_vel_rejected / ekf.n_vel_updates if ekf.n_vel_updates else 0
    print(f"\n[FILTER HEALTH]")
    print(f"  GPS updates rejected:     {ekf.n_gps_rejected}/{ekf.n_gps_updates} ({gps_pct:.1f}%)")
    print(f"  Velocity updates rejected: {ekf.n_vel_rejected}/{ekf.n_vel_updates} ({vel_pct:.1f}%)")
    
    # Summary comparison
    print("\n" + "-"*60)
    print("PERFORMANCE COMPARISON")
    print("-"*60)
    print(f"{'Metric':<30} {'Sync 2':<15} {'Tuned':<15} {'Change':<15}")
    print("-"*75)
    print(f"{'NIS Mean':<30} {3.847:<15.3f} {nis_values.mean():<15.3f} {(nis_values.mean() - 3.847):+.3f}")
    print(f"{'Pre-Corr Mean (m)':<30} {34.82:<15.2f} {pre_errors.mean():<15.2f} {(pre_errors.mean() - 34.82):+.2f}")
    print(f"{'GPS Rejection %':<30} {0.0:<15.1f} {gps_pct:<15.1f} {gps_pct:+.1f}")
    
    return {
        'estimated_xy': estimated_positions,
        'true_xy': np.column_stack([x_m, y_m]),
        'pre_errors': pre_errors,
        'nis_values': nis_values,
        'ekf': ekf
    }

if __name__ == "__main__":
    results = run_tuned_ekf()
    
    # Plot results
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Trajectory
        ax = axes[0, 0]
        true_xy = results['true_xy']
        est_xy = results['estimated_xy']
        ax.plot(true_xy[:, 0], true_xy[:, 1], 'g-', linewidth=1, label='GPS', alpha=0.5)
        ax.plot(est_xy[:, 0], est_xy[:, 1], 'orange', linewidth=0.8, label='EKF Tuned')
        ax.set_title('Final Tuned EKF Trajectory')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.legend()
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        
        # 2. Pre-correction errors
        ax = axes[0, 1]
        pre_errors = results['pre_errors']
        ax.plot(pre_errors, 'b-', linewidth=0.8)
        ax.axhline(pre_errors.mean(), color='r', linestyle='--', label=f'Mean: {pre_errors.mean():.1f}m')
        ax.set_title('Pre-Correction Errors (Tuned)')
        ax.set_xlabel('GPS Update Index')
        ax.set_ylabel('Error (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. NIS Distribution
        ax = axes[1, 0]
        nis_values = results['nis_values']
        ax.hist(nis_values, bins=30, density=True, alpha=0.7, edgecolor='black')
        from scipy.stats import chi2
        x = np.linspace(0, 20, 100)
        ax.plot(x, chi2.pdf(x, df=2), 'r-', linewidth=2, label='chi2(2) PDF')
        ax.axvline(np.mean(nis_values), color='blue', linestyle='--', 
                   label=f'Mean NIS = {np.mean(nis_values):.2f}')
        ax.set_title('NIS Distribution vs chi2(2)')
        ax.set_xlabel('NIS')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Error vs NIS
        ax = axes[1, 1]
        ax.scatter(nis_values, pre_errors, alpha=0.5, s=20)
        ax.axhline(pre_errors.mean(), color='r', linestyle='--', label=f'Mean Error: {pre_errors.mean():.1f}m')
        ax.axvline(2.0, color='g', linestyle='--', label='Target NIS: 2.0')
        ax.set_title('Pre-Correction Error vs NIS')
        ax.set_xlabel('NIS')
        ax.set_ylabel('Error (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tuned_ekf_results.png', dpi=150)
        print("\nPlot saved to: tuned_ekf_results.png")
        
    except Exception as e:
        print(f"\nPlot error: {e}")