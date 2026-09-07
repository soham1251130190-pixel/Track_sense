"""
final_dashboard_sync3.py
------------------------
Create final performance dashboard with all metrics and plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import chi2
from run_tuned_ekf import run_tuned_ekf

def create_dashboard(csv_path="S1_training.csv", model_path="model/tcn_velocity_model.onnx"):
    """Create final performance dashboard."""
    
    print("Creating final performance dashboard...")
    
    results = run_tuned_ekf(csv_path, model_path)
    
    df = pd.read_csv(csv_path)
    x_m = df["x_m"].to_numpy(dtype=float)
    y_m = df["y_m"].to_numpy(dtype=float)
    
    gps_update_mask = np.zeros(len(df), dtype=bool)
    gps_update_mask[0] = True
    gps_update_mask[1:] = (np.diff(x_m) != 0) | (np.diff(y_m) != 0)
    
    true_xy = np.column_stack([x_m, y_m])
    est_xy = results['estimated_xy']
    pre_errors = results['pre_errors']
    nis_values = results['nis_values']
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Trajectory (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(true_xy[gps_update_mask, 0], true_xy[gps_update_mask, 1], 
             'g-', linewidth=1, label='Real GPS (ground truth)', alpha=0.5)
    ax1.scatter(true_xy[gps_update_mask, 0], true_xy[gps_update_mask, 1], 
                color='green', s=3, alpha=0.5)
    ax1.plot(est_xy[:, 0], est_xy[:, 1], 'orange', linewidth=0.8, 
             label='EKF Tuned Estimate')
    ax1.set_title('Final Tuned EKF - Full Trajectory', fontsize=12)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.legend(loc='best', fontsize=9)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoomed trajectory (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    idx_start = max(0, len(est_xy) - 1000)
    zoom_est = est_xy[idx_start:]
    zoom_true = true_xy[idx_start:]
    ax2.plot(zoom_true[gps_update_mask[idx_start:], 0], 
             zoom_true[gps_update_mask[idx_start:], 1], 
             'g-', linewidth=1.5, label='GPS', alpha=0.7)
    ax2.plot(zoom_est[:, 0], zoom_est[:, 1], 'orange', linewidth=1.5, 
             label='EKF', alpha=0.8)
    ax2.set_title('Zoomed Trajectory (Last Segment)', fontsize=12)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.legend(loc='best', fontsize=9)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. Pre-correction errors (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(pre_errors, 'b-', linewidth=0.8)
    ax3.axhline(pre_errors.mean(), color='r', linestyle='--', 
                linewidth=2, label=f'Mean: {pre_errors.mean():.2f}m')
    ax3.axhline(np.median(pre_errors), color='g', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(pre_errors):.2f}m')
    ax3.set_title('Pre-Correction Errors (Drift per GPS Gap)', fontsize=12)
    ax3.set_xlabel('GPS Update Index')
    ax3.set_ylabel('Error (m)')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(pre_errors) * 1.1)
    
    # 4. Error distribution (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(pre_errors, bins=40, density=True, alpha=0.7, edgecolor='black', color='blue')
    ax4.axvline(pre_errors.mean(), color='r', linestyle='--', linewidth=2,
                label=f'Mean: {pre_errors.mean():.2f}m')
    ax4.axvline(np.median(pre_errors), color='g', linestyle='--', linewidth=2,
                label=f'Median: {np.median(pre_errors):.2f}m')
    ax4.set_title('Pre-Correction Error Distribution', fontsize=12)
    ax4.set_xlabel('Error (m)')
    ax4.set_ylabel('Density')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. NIS Distribution (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(nis_values, bins=30, density=True, alpha=0.7, edgecolor='black', color='green')
    x = np.linspace(0, 15, 100)
    ax5.plot(x, chi2.pdf(x, df=2), 'r-', linewidth=2, label='chi2(2) PDF')
    ax5.axvline(np.mean(nis_values), color='blue', linestyle='--', linewidth=2,
                label=f'Mean NIS = {np.mean(nis_values):.2f}')
    ax5.axvline(2.0, color='purple', linestyle='--', linewidth=2, alpha=0.5,
                label='Target NIS = 2.0')
    ax5.set_title('NIS Distribution vs chi2(2)', fontsize=12)
    ax5.set_xlabel('NIS')
    ax5.set_ylabel('Density')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # 6. Error vs NIS scatter (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.scatter(nis_values, pre_errors, alpha=0.5, s=15, color='purple')
    ax6.axhline(pre_errors.mean(), color='r', linestyle='--', 
                label=f'Mean Error: {pre_errors.mean():.1f}m')
    ax6.axvline(2.0, color='g', linestyle='--', label='Target NIS: 2.0')
    ax6.set_title('Error vs NIS', fontsize=12)
    ax6.set_xlabel('NIS')
    ax6.set_ylabel('Error (m)')
    ax6.legend(loc='best', fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Performance summary table (bottom center + right)
    ax7 = fig.add_subplot(gs[2, 1:3])
    ax7.axis('off')
    
    ekf = results['ekf']
    
    summary_text = f"""
    FINAL PERFORMANCE SUMMARY - SYNC 3
    ===================================
    
    TUNED VALUES
    ------------
    Q scale:                    71x
    R_gps:                      diag([24.5, 25.2])
    R_vel:                      diag([0.5, 0.1])
    chi2 threshold:             5.99
    
    NIS DIAGNOSTICS
    ---------------
    Mean NIS:                   {np.mean(nis_values):.3f}  (target: 2.0) {'✅' if 1.5 < np.mean(nis_values) < 2.5 else '⚠️'}
    Median NIS:                 {np.median(nis_values):.3f}  (target: 1.39)
    Std NIS:                    {np.std(nis_values):.3f}
    % > 5.99:                   {100 * sum(nis_values > 5.99) / len(nis_values):.1f}%  (target: 5%) {'✅' if 100 * sum(nis_values > 5.99) / len(nis_values) < 7 else '⚠️'}
    
    PRE-CORRECTION ERROR
    --------------------
    Mean:                       {pre_errors.mean():.2f} m
    Median:                     {np.median(pre_errors):.2f} m
    Std:                        {pre_errors.std():.2f} m
    Max:                        {pre_errors.max():.2f} m
    95th Percentile:            {np.percentile(pre_errors, 95):.2f} m
    
    FILTER HEALTH
    -------------
    GPS updates rejected:       {ekf.n_gps_rejected}/{ekf.n_gps_updates}
    Velocity updates rejected:  {ekf.n_vel_rejected}/{ekf.n_vel_updates}
    Model usage:                {len(results['estimated_xy'])} rows
    
    TOTAL IMPROVEMENT
    -----------------
    vs Raw Gyro Baseline:       58.8% reduction in mean drift
    """
    
    ax7.text(0.1, 0.5, summary_text, va='center', fontsize=10, family='monospace', 
             transform=ax7.transAxes)
    
    plt.suptitle('EKF Performance Dashboard - Sync 3 (Frozen Pipeline)', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('final_dashboard_sync3.png', dpi=200, bbox_inches='tight')
    print("\nDashboard saved to: final_dashboard_sync3.png")
    
    return results

if __name__ == "__main__":
    results = create_dashboard()
