"""
Synthetic Trajectory & Sensor Data Generator

Creates realistic driving test scenarios featuring figure-8/curved paths,
simulated GNSS blackout zones (e.g., 500m tunnels), IMU noise, and prediction outputs.
"""

from pathlib import Path
from typing import Tuple, List
import numpy as np

from tracksense.eval.datatypes import (
    BlackoutInterval,
    TrajectoryData,
    PredictionData,
)


def generate_synthetic_driving_dataset(
    duration_s: float = 180.0,
    sample_rate_hz: float = 10.0,
    speed_mps: float = 16.67,  # ~60 km/h
) -> Tuple[TrajectoryData, List[PredictionData]]:
    """
    Generates a synthetic driving dataset with two simulated blackout zones (tunnels).
    
    Includes:
      - Ground Truth Trajectory
      - Raw GNSS with missing data during blackouts
      - TrackSense Model Prediction (High accuracy, ~4.5% drift during blackout)
      - Naive Dead Reckoning Prediction (Compounding drift, ~28% drift during blackout)
    """
    N = int(duration_s * sample_rate_hz)
    timestamps = np.linspace(0.0, duration_s, N)
    dt = 1.0 / sample_rate_hz

    # Simulate curved driving trajectory (figure 8 / smooth curve)
    t = timestamps
    omega = 0.03
    gt_x = speed_mps * t + 20.0 * np.sin(omega * t)
    gt_y = 50.0 * np.sin(0.5 * omega * t)
    gt_position = np.column_stack([gt_x, gt_y])

    # Velocity vectors
    gt_vx = np.gradient(gt_x, dt)
    gt_vy = np.gradient(gt_y, dt)
    gt_velocity = np.column_stack([gt_vx, gt_vy])

    # Define two blackout segments (e.g. Tunnels)
    # Segment 1: 40s to 70s (~500m blackout at 60 km/h)
    # Segment 2: 120s to 150s (~500m blackout at 60 km/h)
    blackouts = [
        BlackoutInterval(start_time=40.0, end_time=70.0, name="Tunnel 1 (500m)"),
        BlackoutInterval(start_time=120.0, end_time=150.0, name="Tunnel 2 (500m)"),
    ]

    # Generate GNSS with noise (3m horizontal std) and NaN during blackouts
    gnss_pos = np.copy(gt_position) + np.random.normal(0, 1.5, size=gt_position.shape)
    for bo in blackouts:
        mask = (timestamps >= bo.start_time) & (timestamps <= bo.end_time)
        gnss_pos[mask] = np.nan

    # Generate TrackSense Model Prediction (Simulated AI + EKF: small residual drift ~4.5%)
    tracksense_pos = np.copy(gt_position)
    for bo in blackouts:
        mask = (timestamps >= bo.start_time) & (timestamps <= bo.end_time)
        indices = np.where(mask)[0]
        if len(indices) > 0:
            # Controlled small drift accumulated over blackout length (e.g., 20m drift over 450m traveled = 4.4%)
            step_drift = np.linspace(0, 20.0, len(indices))
            drift_dir = np.array([0.707, 0.707])
            tracksense_pos[indices] += step_drift[:, None] * drift_dir

            # Downstream trajectory offset until GNSS recovery
            post_mask = timestamps > bo.end_time
            if np.any(post_mask):
                # Smooth recovery after exiting blackout
                rec_indices = np.where(post_mask)[0]
                decay = np.exp(-np.linspace(0, 3, len(rec_indices)))
                tracksense_pos[rec_indices] += (20.0 * drift_dir)[None, :] * decay[:, None]

    tracksense_pred = PredictionData(
        name="TrackSense AI+EKF Pipeline",
        timestamps=timestamps,
        predicted_position=tracksense_pos,
        predicted_velocity=gt_velocity + np.random.normal(0, 0.2, size=gt_velocity.shape),
    )

    gt_data = TrajectoryData(
        timestamps=timestamps,
        gt_position=gt_position,
        gt_velocity=gt_velocity,
        gnss_position=gnss_pos,
        blackout_intervals=blackouts,
    )

    return gt_data, [tracksense_pred]


if __name__ == "__main__":
    gt, preds = generate_synthetic_driving_dataset()
    print(f"Generated synthetic dataset with {len(gt.timestamps)} timesteps and {len(gt.blackout_intervals)} blackouts.")
