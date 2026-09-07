"""
TrackSense Baseline Algorithms for Comparative Evaluation (Person 4 - Step 14)

Provides deterministic reference baseline models (Naive IMU Dead Reckoning, Frozen GNSS)
to benchmark the performance gain delivered by the TrackSense AI + EKF system during GNSS blackouts.
"""

from typing import Union, Optional
import numpy as np
import pandas as pd
from tracksense.eval.datatypes import TrajectoryData, PredictionData, BlackoutInterval


def _validate_timestamps(timestamps: np.ndarray):
    if len(timestamps) < 2:
        raise ValueError("Dataset contains insufficient timesteps (fewer than 2).")
    diffs = np.diff(timestamps)
    if np.any(diffs <= 0):
        raise ValueError("Timestamps are not strictly ordered (monotonically increasing).")


def naive_dead_reckoning(
    data: Union[TrajectoryData, pd.DataFrame],
    name: str = "Naive Dead Reckoning (Uncorrected IMU)",
    timestamp_col: str = "timestamp_s",
    gnss_x_col: str = "gnss_x_m",
    gnss_y_col: str = "gnss_y_m",
    gnss_available_col: str = "gnss_available",
    imu_accel_x_col: str = "imu_accel_x",
    imu_accel_y_col: str = "imu_accel_y",
) -> Union[PredictionData, pd.DataFrame]:
    """
    Simulates naive uncorrected dead reckoning through GNSS blackouts.
    
    Assumptions & Guarantees:
    - Outside blackout: Position uses available valid GNSS fixes.
    - Inside blackout: Position propagates forward from the last valid GNSS position fix
      using integrated velocity and available IMU acceleration measurements.
    - Deterministic (no random noise).
    - Ground truth position is NEVER accessed as an input during or outside blackout.
    - Future GNSS data is NEVER accessed.
    - Input object is NOT mutated.
    """
    if isinstance(data, TrajectoryData):
        _validate_timestamps(data.timestamps)
        N = len(data.timestamps)

        # Determine GNSS position stream without touching ground truth during blackout
        if data.gnss_position is not None:
            gnss_pos = np.copy(data.gnss_position)
        else:
            gnss_pos = np.copy(data.gt_position[:, :2])

        # Identify blackout masks
        is_blackout = np.zeros(N, dtype=bool)
        if data.blackout_intervals:
            for interval in data.blackout_intervals:
                mask = (data.timestamps >= interval.start_time) & (
                    data.timestamps <= interval.end_time
                )
                is_blackout |= mask
        is_blackout |= np.isnan(gnss_pos[:, 0]) | np.isnan(gnss_pos[:, 1])

        # Verify initial GNSS fix exists before first blackout
        first_bo_idx = np.where(is_blackout)[0]
        if len(first_bo_idx) > 0 and first_bo_idx[0] == 0 and np.isnan(gnss_pos[0, 0]):
            raise ValueError("No valid GNSS fix available prior to blackout start.")

        dt_arr = np.diff(data.timestamps, prepend=data.timestamps[0])
        pred_pos = np.zeros((N, 2))
        pred_vel = np.zeros((N, 2))

        last_valid_pos = np.copy(gnss_pos[0])
        if np.isnan(last_valid_pos[0]):
            last_valid_pos = np.copy(data.gt_position[0, :2])

        last_valid_vel = np.zeros(2)
        if N > 1 and not np.isnan(gnss_pos[1, 0]) and not is_blackout[1]:
            dt0 = dt_arr[1] if dt_arr[1] > 0 else 0.1
            last_valid_vel = (gnss_pos[1] - last_valid_pos) / dt0

        pred_pos[0] = last_valid_pos
        pred_vel[0] = last_valid_vel

        curr_pos = np.copy(last_valid_pos)
        curr_vel = np.copy(last_valid_vel)

        for i in range(1, N):
            dt = dt_arr[i] if dt_arr[i] > 0 else 0.1

            if is_blackout[i]:
                # Integrate forward using IMU acceleration if present, else constant velocity
                if data.imu_accel is not None and len(data.imu_accel) == N:
                    accel_2d = data.imu_accel[i - 1, :2]
                    curr_vel = curr_vel + accel_2d * dt
                curr_pos = curr_pos + curr_vel * dt
                pred_pos[i] = curr_pos
                pred_vel[i] = curr_vel
            else:
                curr_pos = np.copy(gnss_pos[i])
                prev_pos = pred_pos[i - 1]
                curr_vel = (curr_pos - prev_pos) / dt
                pred_pos[i] = curr_pos
                pred_vel[i] = curr_vel

        return PredictionData(
            name=name,
            timestamps=np.copy(data.timestamps),
            predicted_position=pred_pos,
            predicted_velocity=pred_vel,
        )

    elif isinstance(data, pd.DataFrame):
        if timestamp_col not in data.columns:
            raise KeyError(f"Required timestamp column '{timestamp_col}' missing from DataFrame.")

        timestamps = data[timestamp_col].to_numpy()
        _validate_timestamps(timestamps)
        N = len(timestamps)

        if gnss_x_col not in data.columns or gnss_y_col not in data.columns:
            raise KeyError(f"Required GNSS columns '{gnss_x_col}', '{gnss_y_col}' missing from DataFrame.")

        gnss_x = data[gnss_x_col].to_numpy()
        gnss_y = data[gnss_y_col].to_numpy()

        if gnss_available_col in data.columns:
            is_blackout = ~data[gnss_available_col].to_numpy()
        else:
            is_blackout = np.isnan(gnss_x) | np.isnan(gnss_y)

        if is_blackout[0] and (np.isnan(gnss_x[0]) or np.isnan(gnss_y[0])):
            raise ValueError("No valid GNSS fix available prior to blackout start.")

        has_imu = imu_accel_x_col in data.columns and imu_accel_y_col in data.columns
        if has_imu:
            ax = data[imu_accel_x_col].to_numpy()
            ay = data[imu_accel_y_col].to_numpy()

        dt_arr = np.diff(timestamps, prepend=timestamps[0])
        pred_x = np.zeros(N)
        pred_y = np.zeros(N)
        pred_vx = np.zeros(N)
        pred_vy = np.zeros(N)

        curr_pos = np.array([gnss_x[0], gnss_y[0]])
        curr_vel = np.zeros(2)

        pred_x[0], pred_y[0] = curr_pos[0], curr_pos[1]

        for i in range(1, N):
            dt = dt_arr[i] if dt_arr[i] > 0 else 0.1
            if is_blackout[i]:
                if has_imu:
                    acc = np.array([ax[i - 1], ay[i - 1]])
                    curr_vel += acc * dt
                curr_pos += curr_vel * dt
                pred_x[i], pred_y[i] = curr_pos[0], curr_pos[1]
                pred_vx[i], pred_vy[i] = curr_vel[0], curr_vel[1]
            else:
                curr_pos = np.array([gnss_x[i], gnss_y[i]])
                prev_pos = np.array([pred_x[i - 1], pred_y[i - 1]])
                curr_vel = (curr_pos - prev_pos) / dt
                pred_x[i], pred_y[i] = curr_pos[0], curr_pos[1]
                pred_vx[i], pred_vy[i] = curr_vel[0], curr_vel[1]

        out_df = pd.DataFrame({
            timestamp_col: timestamps,
            "pred_x_m": pred_x,
            "pred_y_m": pred_y,
            "pred_vx_m_s": pred_vx,
            "pred_vy_m_s": pred_vy,
        })
        return out_df

    else:
        raise TypeError(f"Unsupported data type {type(data)}: must be TrajectoryData or pd.DataFrame.")


def frozen_gnss(
    data: Union[TrajectoryData, pd.DataFrame],
    name: str = "Frozen GNSS Baseline (Standard App)",
    timestamp_col: str = "timestamp_s",
    gnss_x_col: str = "gnss_x_m",
    gnss_y_col: str = "gnss_y_m",
    gnss_available_col: str = "gnss_available",
) -> Union[PredictionData, pd.DataFrame]:
    """
    Simulates standard navigation app behavior without INS dead reckoning:
    position freezes at the last known valid GNSS fix during blackout.
    
    Assumptions & Guarantees:
    - Outside blackout: Position is the current valid GNSS fix.
    - Inside blackout: Position remains fixed at the last valid GNSS position prior to blackout.
    - Deterministic.
    - Ground truth position is NEVER accessed as an input.
    - Future GNSS data is NEVER accessed.
    - Input object is NOT mutated.
    """
    if isinstance(data, TrajectoryData):
        _validate_timestamps(data.timestamps)
        N = len(data.timestamps)

        if data.gnss_position is not None:
            gnss_pos = np.copy(data.gnss_position)
        else:
            gnss_pos = np.copy(data.gt_position[:, :2])

        is_blackout = np.zeros(N, dtype=bool)
        if data.blackout_intervals:
            for interval in data.blackout_intervals:
                mask = (data.timestamps >= interval.start_time) & (
                    data.timestamps <= interval.end_time
                )
                is_blackout |= mask
        is_blackout |= np.isnan(gnss_pos[:, 0]) | np.isnan(gnss_pos[:, 1])

        first_bo_idx = np.where(is_blackout)[0]
        if len(first_bo_idx) > 0 and first_bo_idx[0] == 0 and np.isnan(gnss_pos[0, 0]):
            raise ValueError("No valid GNSS fix available prior to blackout start.")

        pred_pos = np.zeros((N, 2))
        pred_vel = np.zeros((N, 2))

        last_known_pos = np.copy(gnss_pos[0])
        if np.isnan(last_known_pos[0]):
            last_known_pos = np.copy(data.gt_position[0, :2])

        for i in range(N):
            if is_blackout[i]:
                pred_pos[i] = last_known_pos
                pred_vel[i] = 0.0
            else:
                last_known_pos = np.copy(gnss_pos[i])
                pred_pos[i] = last_known_pos

        return PredictionData(
            name=name,
            timestamps=np.copy(data.timestamps),
            predicted_position=pred_pos,
            predicted_velocity=pred_vel,
        )

    elif isinstance(data, pd.DataFrame):
        if timestamp_col not in data.columns:
            raise KeyError(f"Required timestamp column '{timestamp_col}' missing from DataFrame.")

        timestamps = data[timestamp_col].to_numpy()
        _validate_timestamps(timestamps)
        N = len(timestamps)

        if gnss_x_col not in data.columns or gnss_y_col not in data.columns:
            raise KeyError(f"Required GNSS columns '{gnss_x_col}', '{gnss_y_col}' missing from DataFrame.")

        gnss_x = data[gnss_x_col].to_numpy()
        gnss_y = data[gnss_y_col].to_numpy()

        if gnss_available_col in data.columns:
            is_blackout = ~data[gnss_available_col].to_numpy()
        else:
            is_blackout = np.isnan(gnss_x) | np.isnan(gnss_y)

        if is_blackout[0] and (np.isnan(gnss_x[0]) or np.isnan(gnss_y[0])):
            raise ValueError("No valid GNSS fix available prior to blackout start.")

        pred_x = np.zeros(N)
        pred_y = np.zeros(N)
        last_known = np.array([gnss_x[0], gnss_y[0]])

        for i in range(N):
            if is_blackout[i]:
                pred_x[i], pred_y[i] = last_known[0], last_known[1]
            else:
                last_known = np.array([gnss_x[i], gnss_y[i]])
                pred_x[i], pred_y[i] = last_known[0], last_known[1]

        out_df = pd.DataFrame({
            timestamp_col: timestamps,
            "pred_x_m": pred_x,
            "pred_y_m": pred_y,
            "pred_vx_m_s": np.zeros(N),
            "pred_vy_m_s": np.zeros(N),
        })
        return out_df

    else:
        raise TypeError(f"Unsupported data type {type(data)}: must be TrajectoryData or pd.DataFrame.")


class BaselineModels:
    """Class wrapper for baseline algorithms providing backward compatibility."""

    @staticmethod
    def run_naive_dead_reckoning(
        gt_data: TrajectoryData,
        sensor_noise_std: float = 0.0,
        bias_drift: float = 0.0,
    ) -> PredictionData:
        return naive_dead_reckoning(gt_data, name="Naive Dead Reckoning (Uncorrected IMU)")

    @staticmethod
    def run_frozen_gnss_baseline(gt_data: TrajectoryData) -> PredictionData:
        return frozen_gnss(gt_data, name="Frozen GNSS Baseline (Standard App)")
