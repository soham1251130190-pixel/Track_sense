"""
TrackSense Real-Data Evaluation Plotting Generator

Runs official real-data evaluation pipeline dynamically and generates 5 publication-ready figures:
1. Ground-truth XY trajectory vs TCN+EKF estimated XY trajectory (with GNSS blackout zones).
2. Euclidean position error over time (with shaded 30s, 60s, 120s blackout intervals).
3. Blackout drift percentage by blackout duration vs 10% target threshold.
4. Final position error by blackout duration.
5. Ground-truth speed vs TCN predicted speed over time (with sparse GPS limitation note).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure src and root are in PYTHONPATH
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))
sys.path.insert(0, str(root_dir))

from tracksense.eval.datatypes import TrajectoryData, BlackoutInterval
from tracksense.eval.adapters import TCN_EKF_ModelAdapter
from tracksense.eval.harness import EvaluationHarness
from tracksense.eval.metrics import EvaluationMetrics


def run_evaluation_and_generate_plots():
    # ---------------------------------------------------------
    # 1. Load Data & Prepare Trajectory Data
    # ---------------------------------------------------------
    csv_path = root_dir / "data" / "S1_training.csv"
    if not csv_path.exists():
        csv_path = root_dir / "model" / "S1_training.csv"

    print(f"Loading evaluation dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Use first 3000 samples (300 seconds at 10Hz)
    N = min(len(df), 3000)
    df = df.iloc[:N]

    timestamps = np.arange(len(df)) * 0.1
    gt_pos = np.column_stack([df["x_m"].values, df["y_m"].values]).astype(float)
    csv_speed_mps = (df["speed_kmh"].values / 3.6).astype(float)

    vx = np.gradient(gt_pos[:, 0], timestamps)
    vy = np.gradient(gt_pos[:, 1], timestamps)
    gt_vel = np.column_stack([vx, vy])

    imu_accel = np.column_stack([df["accel_x"].values, df["accel_y"].values, df["accel_z"].values]).astype(float)
    imu_gyro = np.column_stack([df["gyro_yaw"].values, df["gyro_pitch"].values, df["gyro_roll"].values]).astype(float)

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

    # ---------------------------------------------------------
    # 2. Run TCN+EKF Pipeline dynamically
    # ---------------------------------------------------------
    print("Running TrackSense TCN+EKF adapter forward pass...")
    adapter = TCN_EKF_ModelAdapter(checkpoint_path=str(root_dir / "model" / "best_model_real.pt"))
    pred_tracksense = adapter.run_pipeline(gt_data)

    # Extract raw TCN model output predictions directly for speed comparison plot
    imu_matrix = adapter._extract_imu_matrix_from_trajectory(gt_data)
    windows_tensor = adapter._build_sliding_windows(imu_matrix)
    import torch
    with torch.no_grad():
        tcn_out = adapter.model(windows_tensor).cpu().numpy()
    tcn_speed_pred_mps = tcn_out[:, 0]

    # Run Harness dynamically
    harness = EvaluationHarness(target_drift_pct=10.0)
    out_dir = root_dir / "eval_real_fixed_results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = harness.run_evaluation(
        gt_data=gt_data,
        predictions=[pred_tracksense],
        output_dir=str(root_dir / "eval_real_fixed_results"),
        include_baselines=True
    )

    metrics_ts, status_ts = results["TrackSense AI + EKF Pipeline"]

    # Plotting setup
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

    generated_files = []

    # ---------------------------------------------------------
    # Figure 1: 2D Spatial Trajectory Map
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 8), dpi=200)

    # Ground Truth
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], label="Ground Truth Trajectory", color="black", linewidth=2.5, zorder=5)

    # TCN+EKF Estimation
    ax1.plot(
        pred_tracksense.predicted_position[:, 0],
        pred_tracksense.predicted_position[:, 1],
        label="TrackSense TCN+EKF Estimate",
        color="#D55E00",
        linewidth=2.0,
        linestyle="--",
        zorder=6
    )

    # GNSS Measurements outside blackouts
    valid_gnss = ~np.isnan(gnss_pos[:, 0])
    ax1.scatter(
        gnss_pos[valid_gnss, 0],
        gnss_pos[valid_gnss, 1],
        label="Available GNSS Fixes",
        color="#0072B2",
        s=10,
        alpha=0.5,
        zorder=3
    )

    # Highlight Blackout Intervals
    first_b = True
    for b in blackouts:
        mask = (timestamps >= b.start_time) & (timestamps <= b.end_time)
        b_gt = gt_pos[mask]
        label = "GNSS Blackout Segment" if first_b else ""
        ax1.plot(b_gt[:, 0], b_gt[:, 1], color="#CC79A7", linewidth=5.0, alpha=0.5, label=label, zorder=4)
        first_b = False

    ax1.set_title("Ground-Truth vs TCN+EKF Estimated Trajectory (Real Data S1)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("East / X Position (m)", fontsize=11)
    ax1.set_ylabel("North / Y Position (m)", fontsize=11)
    ax1.axis("equal")
    ax1.legend(loc="best", frameon=True, facecolor="white")
    plt.tight_layout()

    f1_path = out_dir / "fig1_xy_trajectory.png"
    plt.savefig(f1_path)
    plt.close(fig1)
    generated_files.append(str(f1_path))

    # ---------------------------------------------------------
    # Figure 2: Position Error Over Time
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(12, 6), dpi=200)

    errors_m = EvaluationMetrics.compute_position_errors(gt_pos, pred_tracksense.predicted_position)

    # Shade blackout windows
    colors_b = ["#FFCCCC", "#FFE5CC", "#E5CCFF"]
    for idx, b in enumerate(blackouts):
        dur_s = b.end_time - b.start_time
        ax2.axvspan(b.start_time, b.end_time, color=colors_b[idx % len(colors_b)], alpha=0.5, zorder=1)
        # Add text box over blackout window
        mid_t = (b.start_time + b.end_time) / 2.0
        ax2.text(
            mid_t,
            np.max(errors_m) * 0.85,
            f"{b.name}\n({dur_s:.0f}s duration)",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#999999")
        )

    ax2.plot(timestamps, errors_m, label=f"TrackSense TCN+EKF (RMSE: {metrics_ts.overall_rmse_m:.1f} m)", color="#D55E00", linewidth=2.0, zorder=3)

    ax2.set_title("Euclidean Position Error Over Time Across Blackout Intervals", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Timestamp (seconds)", fontsize=11)
    ax2.set_ylabel("Position Error (meters)", fontsize=11)
    ax2.legend(loc="upper left", frameon=True, facecolor="white")
    plt.tight_layout()

    f2_path = out_dir / "fig2_position_error_time.png"
    plt.savefig(f2_path)
    plt.close(fig2)
    generated_files.append(str(f2_path))

    # ---------------------------------------------------------
    # Figure 3: Blackout Drift Percentage by Duration
    # ---------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(9, 6), dpi=200)

    seg_names = [f"{sm.blackout_duration_s:.0f}s" for sm in metrics_ts.segment_metrics]
    drift_pcts = [sm.drift_percentage for sm in metrics_ts.segment_metrics]

    bars3 = ax3.bar(seg_names, drift_pcts, width=0.45, color=["#009E73" if d <= 10.0 else "#D55E00" for d in drift_pcts], edgecolor="#333333", zorder=3)

    # 10% Target Line
    ax3.axhline(10.0, color="red", linestyle="--", linewidth=2.0, label="Project Target Ceiling (<10% Drift)", zorder=5)

    ax3.set_title("Dead-Reckoning Drift Percentage by Blackout Duration", fontsize=13, fontweight="bold", pad=12)
    ax3.set_xlabel("Blackout Duration", fontsize=11)
    ax3.set_ylabel("Drift Percentage (%)", fontsize=11)
    ax3.set_ylim(0, max(max(drift_pcts) * 1.2, 15.0))
    ax3.legend(loc="upper left", frameon=True, facecolor="white")

    # Annotate bars
    for bar in bars3:
        height = bar.get_height()
        ax3.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    f3_path = out_dir / "fig3_drift_percentage.png"
    plt.savefig(f3_path)
    plt.close(fig3)
    generated_files.append(str(f3_path))

    # ---------------------------------------------------------
    # Figure 4: Final Position Error by Blackout Duration
    # ---------------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(9, 6), dpi=200)

    end_errors_m = [sm.end_drift_distance_m for sm in metrics_ts.segment_metrics]

    bars4 = ax4.bar(seg_names, end_errors_m, width=0.45, color="#0072B2", edgecolor="#333333", zorder=3)

    ax4.set_title("Final Position Error at End of Blackout Intervals", fontsize=13, fontweight="bold", pad=12)
    ax4.set_xlabel("Blackout Duration", fontsize=11)
    ax4.set_ylabel("Final Position Error (meters)", fontsize=11)
    ax4.set_ylim(0, max(end_errors_m) * 1.2)

    # Annotate bars
    for bar in bars4:
        height = bar.get_height()
        ax4.annotate(
            f"{height:.2f} m",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    f4_path = out_dir / "fig4_final_position_error.png"
    plt.savefig(f4_path)
    plt.close(fig4)
    generated_files.append(str(f4_path))

    # ---------------------------------------------------------
    # Figure 5: Ground-Truth Speed vs TCN Predicted Speed Over Time
    # ---------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(12, 6), dpi=200)

    # GT CSV Speed
    ax5.plot(timestamps, csv_speed_mps, label="GT CSV Speed (speed_kmh / 3.6)", color="black", linewidth=1.5, alpha=0.7, zorder=3)

    # TCN Model Speed Prediction
    ax5.plot(timestamps, tcn_speed_pred_mps, label="TCN Predicted Speed (m/s)", color="#D55E00", linewidth=2.0, linestyle="--", zorder=4)

    # EKF Fused Speed Estimate
    ekf_speed_mps = np.linalg.norm(pred_tracksense.predicted_velocity, axis=1)
    ax5.plot(timestamps, ekf_speed_mps, label="EKF Speed Estimate (m/s)", color="#009E73", linewidth=1.5, alpha=0.8, zorder=5)

    # Highlight Blackout Intervals
    first_b = True
    for b in blackouts:
        label = "GNSS Blackout Period" if first_b else ""
        ax5.axvspan(b.start_time, b.end_time, color="#FF9999", alpha=0.3, label=label, zorder=1)
        first_b = False

    # Callout Note on Sparse GT GPS Limitation
    ax5.text(
        0.02,
        0.95,
        "Note: GT speed column (speed_kmh/3.6) exhibits stepped artifacts\ndue to sparse ~0.11 Hz (9s) GPS position updates in dataset.",
        transform=ax5.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFFCC", alpha=0.9, edgecolor="#CCCC99")
    )

    ax5.set_title("Ground-Truth Speed vs TCN Predicted Speed Over Time", fontsize=13, fontweight="bold", pad=12)
    ax5.set_xlabel("Timestamp (seconds)", fontsize=11)
    ax5.set_ylabel("Speed (m/s)", fontsize=11)
    ax5.legend(loc="upper right", frameon=True, facecolor="white")
    plt.tight_layout()

    f5_path = out_dir / "fig5_speed_comparison.png"
    plt.savefig(f5_path)
    plt.close(fig5)
    generated_files.append(str(f5_path))

    print("\nSuccessfully generated evaluation figures:")
    for filepath in generated_files:
        print(f"  - {filepath}")

    return generated_files


if __name__ == "__main__":
    run_evaluation_and_generate_plots()
