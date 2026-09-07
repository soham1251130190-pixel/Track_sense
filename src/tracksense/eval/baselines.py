"""
TrackSense Baseline Algorithms for Comparative Evaluation

Provides standard non-AI baseline dead-reckoning models (Naive Integration, Frozen GNSS)
to benchmark the performance gain delivered by TrackSense AI + EKF.
"""

from typing import Optional
import numpy as np
from tracksense.eval.datatypes import TrajectoryData, PredictionData


class BaselineModels:
    """Generates baseline benchmark predictions for evaluation comparison."""

    @staticmethod
    def run_naive_dead_reckoning(
        gt_data: TrajectoryData,
        sensor_noise_std: float = 0.05,
        bias_drift: float = 0.02,
    ) -> PredictionData:
        """
        Simulates naive uncorrected inertial dead reckoning (double integration of IMU).
        
        Demonstrates compounding exponential/quadratic position drift during GNSS blackouts.
        """
        N = len(gt_data.timestamps)
        pred_pos = np.copy(gt_data.gt_position[:, :2])
        pred_vel = (
            np.copy(gt_data.gt_velocity[:, :2])
            if gt_data.gt_velocity is not None
            else np.zeros((N, 2))
        )

        dt_arr = np.diff(gt_data.timestamps, prepend=gt_data.timestamps[0])

        is_blackout = np.zeros(N, dtype=bool)
        for interval in gt_data.blackout_intervals:
            mask = (gt_data.timestamps >= interval.start_time) & (
                gt_data.timestamps <= interval.end_time
            )
            is_blackout |= mask

        # Simulate uncorrected drift during blackout periods
        current_pos = np.copy(pred_pos[0])
        current_vel = np.copy(pred_vel[0])
        accumulated_bias = np.array([bias_drift, bias_drift])

        for i in range(1, N):
            dt = dt_arr[i]
            if dt <= 0:
                dt = 0.1

            if is_blackout[i]:
                # Add noisy acceleration integration and compounding bias drift
                if gt_data.gt_velocity is not None:
                    true_vel = gt_data.gt_velocity[i, :2]
                else:
                    true_vel = current_vel

                # Accelerometer error accumulation
                noise = np.random.normal(0, sensor_noise_std, size=2)
                accumulated_bias += np.array([bias_drift * dt, bias_drift * dt])
                err_accel = noise + accumulated_bias

                current_vel = true_vel + err_accel * (i * dt * 0.1)
                current_pos = current_pos + current_vel * dt
                pred_pos[i] = current_pos
                pred_vel[i] = current_vel
            else:
                # Reset to true state when GNSS is available
                current_pos = gt_data.gt_position[i, :2]
                if gt_data.gt_velocity is not None:
                    current_vel = gt_data.gt_velocity[i, :2]
                accumulated_bias = np.array([bias_drift, bias_drift])
                pred_pos[i] = current_pos
                pred_vel[i] = current_vel

        return PredictionData(
            name="Naive Dead Reckoning (Uncorrected IMU)",
            timestamps=gt_data.timestamps,
            predicted_position=pred_pos,
            predicted_velocity=pred_vel,
        )

    @staticmethod
    def run_frozen_gnss_baseline(gt_data: TrajectoryData) -> PredictionData:
        """
        Simulates standard navigation app behavior without INS dead reckoning:
        position freezes at last known GNSS fix during a blackout, then snaps.
        """
        N = len(gt_data.timestamps)
        pred_pos = np.copy(gt_data.gt_position[:, :2])
        pred_vel = np.zeros((N, 2))

        is_blackout = np.zeros(N, dtype=bool)
        for interval in gt_data.blackout_intervals:
            mask = (gt_data.timestamps >= interval.start_time) & (
                gt_data.timestamps <= interval.end_time
            )
            is_blackout |= mask

        last_known_pos = np.copy(pred_pos[0])

        for i in range(N):
            if is_blackout[i]:
                pred_pos[i] = last_known_pos
                pred_vel[i] = 0.0
            else:
                last_known_pos = gt_data.gt_position[i, :2]
                pred_pos[i] = last_known_pos
                if gt_data.gt_velocity is not None:
                    pred_vel[i] = gt_data.gt_velocity[i, :2]

        return PredictionData(
            name="Frozen GNSS Baseline (Standard App)",
            timestamps=gt_data.timestamps,
            predicted_position=pred_pos,
            predicted_velocity=pred_vel,
        )
