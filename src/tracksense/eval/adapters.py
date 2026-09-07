"""
TrackSense Model Adapter (Step 15 Integration)

Connects the real trained VelocityTCN model and VehicleEKF fusion pipeline
to the TrackSense evaluation harness (tracksense.eval).
"""

from pathlib import Path
from typing import Union, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch

try:
    from tracksense.models.tcn_model import VelocityTCN
except ImportError:
    from model.model import VelocityTCN

try:
    from tracksense.fusion.ekf_fusion import VehicleEKF
except ImportError:
    from ekf_fusion import VehicleEKF

from tracksense.eval.datatypes import (
    TrajectoryData,
    PredictionData,
    BlackoutInterval,
)


def _validate_timestamps(timestamps: np.ndarray):
    if len(timestamps) < 2:
        raise ValueError("Dataset contains insufficient timesteps (fewer than 2).")
    diffs = np.diff(timestamps)
    if np.any(diffs <= 0):
        raise ValueError("Timestamps are not strictly ordered (monotonically increasing).")


class TCN_EKF_ModelAdapter:
    """
    Adapter connecting VelocityTCN and VehicleEKF to the tracksense.eval framework.
    
    Data Flow:
        Input trajectory / DataFrame (IMU sensor streams)
            ↓
        100-sample sliding windows (shape: [batch, 6, 100])
            ↓
        VelocityTCN forward pass (checkpoint: model/best_model_real.pt)
            ↓
        Predictions: [speed_mean (km/h), speed_log_var, heading_rate (rad/s)]
            ↓
        VehicleEKF fusion (with GNSS position updates outside blackout,
                           TCN dead reckoning ONLY inside blackout)
            ↓
        Estimated trajectory (PredictionData or pd.DataFrame)
    """

    def __init__(
        self,
        checkpoint_path: str = "model/best_model_real.pt",
        device: Optional[str] = None,
        name: str = "TrackSense AI + EKF Pipeline",
        window_size: int = 100,
        model_instance: Optional[torch.nn.Module] = None,
    ):
        self.name = name
        self.window_size = window_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if model_instance is not None:
            self.model = model_instance.to(self.device)
            self.model.eval()
        else:
            ckpt_path = Path(checkpoint_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Model checkpoint not found at '{checkpoint_path}'. "
                    "Ensure best_model_real.pt is placed in the model/ directory."
                )

            self.model = VelocityTCN(in_ch=6, channels=64, n_layers=5).to(self.device)
            checkpoint = torch.load(str(ckpt_path), map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()

    def predict(
        self,
        data: Union[TrajectoryData, pd.DataFrame],
        name: Optional[str] = None,
    ) -> Union[PredictionData, pd.DataFrame]:
        """
        Executes end-to-end model inference and EKF state estimation.
        
        Args:
            data: Input dataset (TrajectoryData or pd.DataFrame)
            name: Optional override for prediction name
            
        Returns:
            PredictionData object if input is TrajectoryData, or pd.DataFrame if input is pd.DataFrame
        """
        pred_name = name or self.name

        if isinstance(data, TrajectoryData):
            return self._predict_trajectory_data(data, name=pred_name)
        elif isinstance(data, pd.DataFrame):
            return self._predict_dataframe(data, name=pred_name)
        else:
            raise TypeError(
                f"Unsupported data type {type(data)}: must be TrajectoryData or pd.DataFrame."
            )

    run_pipeline = predict

    def _extract_imu_matrix_from_trajectory(self, data: TrajectoryData) -> np.ndarray:
        """
        Extracts 6-channel IMU matrix of shape (N, 6) in required channel order:
        [accel_x, accel_y, accel_z, gyro_yaw, gyro_pitch, gyro_roll]
        """
        N = len(data.timestamps)
        if data.imu_accel is None or data.imu_gyro is None:
            raise ValueError(
                "TrajectoryData must contain 'imu_accel' (N, 3) and 'imu_gyro' (N, 3)."
            )

        if len(data.imu_accel) != N or len(data.imu_gyro) != N:
            raise ValueError("IMU sensor array lengths do not match timestamps count.")

        # Accel 3D
        ax = data.imu_accel[:, 0]
        ay = data.imu_accel[:, 1]
        az = data.imu_accel[:, 2]

        # Gyro 3D: input data.imu_gyro channel order is [yaw, pitch, roll]
        gz = data.imu_gyro[:, 0]
        gy = data.imu_gyro[:, 1] if data.imu_gyro.shape[1] > 1 else np.zeros(N)
        gx = data.imu_gyro[:, 2] if data.imu_gyro.shape[1] > 2 else np.zeros(N)

        imu_matrix = np.column_stack([ax, ay, az, gz, gy, gx])
        return imu_matrix

    def _extract_imu_matrix_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts 6-channel IMU matrix of shape (N, 6) from DataFrame columns.
        """
        required_cols = [
            "accel_x", "accel_y", "accel_z",
            "gyro_yaw", "gyro_pitch", "gyro_roll"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"DataFrame is missing required IMU columns: {missing}")

        imu_matrix = df[required_cols].to_numpy(dtype=np.float32)
        return imu_matrix

    def _build_sliding_windows(self, imu_matrix: np.ndarray) -> torch.Tensor:
        """
        Constructs (N, 6, window_size) tensor for all N timesteps.
        Pads initial timesteps < window_size by repeating index 0.
        """
        N, num_channels = imu_matrix.shape
        if num_channels != 6:
            raise ValueError(f"Expected 6 IMU channels, got {num_channels}.")

        windows = np.zeros((N, 6, self.window_size), dtype=np.float32)
        for i in range(N):
            if i + 1 < self.window_size:
                # Pad start of sequence by prepending first sample
                pad_len = self.window_size - (i + 1)
                chunk = imu_matrix[: i + 1]  # (i+1, 6)
                first_row = imu_matrix[0:1]  # (1, 6)
                padded_chunk = np.vstack([np.repeat(first_row, pad_len, axis=0), chunk])
                windows[i] = padded_chunk.T
            else:
                chunk = imu_matrix[i + 1 - self.window_size : i + 1]  # (100, 6)
                windows[i] = chunk.T

        return torch.from_numpy(windows).to(self.device)

    def _predict_trajectory_data(
        self, data: TrajectoryData, name: str
    ) -> PredictionData:
        _validate_timestamps(data.timestamps)
        N = len(data.timestamps)
        if N < self.window_size:
            raise ValueError(
                f"Trajectory length ({N}) is smaller than required window size ({self.window_size})."
            )

        # 1. Extract IMU matrix (N, 6)
        imu_matrix = self._extract_imu_matrix_from_trajectory(data)

        # 2. Extract GNSS availability and position stream without touching ground truth during blackout
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

        # 3. TCN Forward Pass
        windows_tensor = self._build_sliding_windows(imu_matrix)
        with torch.no_grad():
            tcn_output = self.model(windows_tensor).cpu().numpy()  # (N, 3)

        # TCN prediction channel 0 is numerically in m/s for real data
        speed_mean_mps = tcn_output[:, 0]
        speed_log_var = tcn_output[:, 1]
        heading_rate = tcn_output[:, 2]

        # 4. EKF Integration
        dt_arr = np.diff(data.timestamps, prepend=data.timestamps[0])

        # Find initial GNSS fix prior to blackout
        valid_gnss_idx = np.where(~is_blackout)[0]
        if len(valid_gnss_idx) > 0:
            init_idx = valid_gnss_idx[0]
            init_x, init_y = gnss_pos[init_idx, 0], gnss_pos[init_idx, 1]
        else:
            init_x, init_y = 0.0, 0.0

        # Estimate initial heading from first meaningful displacement (>0.5 m) prior to blackout
        init_heading = 0.0
        if len(valid_gnss_idx) >= 2:
            base_pos = gnss_pos[valid_gnss_idx[0]]
            for idx in valid_gnss_idx[1:]:
                dx = gnss_pos[idx, 0] - base_pos[0]
                dy = gnss_pos[idx, 1] - base_pos[1]
                dist = np.hypot(dx, dy)
                if dist > 0.5:
                    init_heading = np.arctan2(dy, dx)
                    break

        init_v_mps = speed_mean_mps[0]
        ekf = VehicleEKF(
            initial_state=[init_x, init_y, init_v_mps, init_heading]
        )

        pred_pos = np.zeros((N, 2))
        pred_vel = np.zeros((N, 2))
        curr_heading = init_heading

        for i in range(N):
            dt = dt_arr[i] if dt_arr[i] > 0 else 0.1

            # Before entering GNSS blackout, derive current heading from latest pre-blackout GNSS displacement >0.5m
            if is_blackout[i] and (i == 0 or not is_blackout[i - 1]):
                prev_valid = np.where(~is_blackout[:i])[0]
                if len(prev_valid) >= 2:
                    latest_idx = prev_valid[-1]
                    latest_pos = gnss_pos[latest_idx]
                    for p_idx in reversed(prev_valid[:-1]):
                        dx = latest_pos[0] - gnss_pos[p_idx, 0]
                        dy = latest_pos[1] - gnss_pos[p_idx, 1]
                        dist = np.hypot(dx, dy)
                        if dist > 0.5:
                            curr_heading = np.arctan2(dy, dx)
                            ekf.x[3] = curr_heading
                            break

            ekf.predict(dt)
            curr_heading += imu_matrix[i, 3] * dt

            # TCN speed output is in m/s; convert m/s -> km/h exactly once so VehicleEKF's internal /3.6 converts it back to m/s
            v_kmh_for_ekf = float(speed_mean_mps[i] * 3.6)
            ekf.update_velocity(
                v_kmh=v_kmh_for_ekf,
                heading_meas=curr_heading,
                velocity_log_variance=speed_log_var[i],
            )

            # Strictly NO DATA LEAKAGE: GNSS updates fed ONLY outside blackout
            if not is_blackout[i]:
                ekf.update_gps(gnss_pos[i, 0], gnss_pos[i, 1])

            pred_pos[i] = ekf.x[:2]
            v_est = ekf.x[2]
            h_est = ekf.x[3]
            pred_vel[i] = [v_est * np.cos(h_est), v_est * np.sin(h_est)]

        return PredictionData(
            name=name,
            timestamps=np.copy(data.timestamps),
            predicted_position=pred_pos,
            predicted_velocity=pred_vel,
            predicted_log_variance=speed_log_var,
        )

    def _predict_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        ts_col = "timestamp" if "timestamp" in df.columns else "timestamp_s"
        if ts_col not in df.columns:
            raise KeyError(f"Required timestamp column missing from DataFrame.")

        timestamps = df[ts_col].to_numpy()
        _validate_timestamps(timestamps)
        N = len(timestamps)
        if N < self.window_size:
            raise ValueError(
                f"DataFrame length ({N}) is smaller than required window size ({self.window_size})."
            )

        # 1. IMU Matrix
        imu_matrix = self._extract_imu_matrix_from_dataframe(df)

        # 2. GNSS position & availability
        gnss_x_col = "gnss_x" if "gnss_x" in df.columns else ("gnss_x_m" if "gnss_x_m" in df.columns else "x_m")
        gnss_y_col = "gnss_y" if "gnss_y" in df.columns else ("gnss_y_m" if "gnss_y_m" in df.columns else "y_m")

        gnss_x = df[gnss_x_col].to_numpy() if gnss_x_col in df.columns else np.full(N, np.nan)
        gnss_y = df[gnss_y_col].to_numpy() if gnss_y_col in df.columns else np.full(N, np.nan)

        if "gnss_available" in df.columns:
            is_blackout = ~df["gnss_available"].to_numpy(dtype=bool)
        else:
            is_blackout = np.isnan(gnss_x) | np.isnan(gnss_y)

        # 3. TCN Forward Pass
        windows_tensor = self._build_sliding_windows(imu_matrix)
        with torch.no_grad():
            tcn_output = self.model(windows_tensor).cpu().numpy()

        speed_mean_mps = tcn_output[:, 0]
        speed_log_var = tcn_output[:, 1]
        heading_rate = tcn_output[:, 2]

        # 4. EKF Integration
        dt_arr = np.diff(timestamps, prepend=timestamps[0])
        valid_gnss_idx = np.where(~is_blackout)[0]
        if len(valid_gnss_idx) > 0:
            init_idx = valid_gnss_idx[0]
            init_x, init_y = gnss_x[init_idx], gnss_y[init_idx]
        else:
            init_x, init_y = 0.0, 0.0

        init_heading = 0.0
        if len(valid_gnss_idx) >= 2:
            base_x, base_y = gnss_x[valid_gnss_idx[0]], gnss_y[valid_gnss_idx[0]]
            for idx in valid_gnss_idx[1:]:
                dx = gnss_x[idx] - base_x
                dy = gnss_y[idx] - base_y
                dist = np.hypot(dx, dy)
                if dist > 0.5:
                    init_heading = np.arctan2(dy, dx)
                    break

        init_v_mps = speed_mean_mps[0]
        ekf = VehicleEKF(initial_state=[init_x, init_y, init_v_mps, init_heading])

        pred_x = np.zeros(N)
        pred_y = np.zeros(N)
        pred_vx = np.zeros(N)
        pred_vy = np.zeros(N)
        curr_heading = init_heading

        for i in range(N):
            dt = dt_arr[i] if dt_arr[i] > 0 else 0.1

            if is_blackout[i] and (i == 0 or not is_blackout[i - 1]):
                prev_valid = np.where(~is_blackout[:i])[0]
                if len(prev_valid) >= 2:
                    latest_idx = prev_valid[-1]
                    latest_pos = np.array([gnss_x[latest_idx], gnss_y[latest_idx]])
                    for p_idx in reversed(prev_valid[:-1]):
                        dx = latest_pos[0] - gnss_x[p_idx]
                        dy = latest_pos[1] - gnss_y[p_idx]
                        dist = np.hypot(dx, dy)
                        if dist > 0.5:
                            curr_heading = np.arctan2(dy, dx)
                            ekf.x[3] = curr_heading
                            break

            ekf.predict(dt)
            curr_heading += imu_matrix[i, 3] * dt
            v_kmh_for_ekf = float(speed_mean_mps[i] * 3.6)
            ekf.update_velocity(
                v_kmh=v_kmh_for_ekf,
                heading_meas=curr_heading,
                velocity_log_variance=speed_log_var[i],
            )

            # Strictly NO DATA LEAKAGE: GNSS updates fed ONLY outside blackout
            if not is_blackout[i]:
                ekf.update_gps(gnss_x[i], gnss_y[i])

            pred_x[i], pred_y[i] = ekf.x[0], ekf.x[1]
            v_est = ekf.x[2]
            h_est = ekf.x[3]
            pred_vx[i], pred_vy[i] = v_est * np.cos(h_est), v_est * np.sin(h_est)

        out_df = pd.DataFrame({
            ts_col: timestamps,
            "pred_x_m": pred_x,
            "pred_y_m": pred_y,
            "pred_vx_m_s": pred_vx,
            "pred_vy_m_s": pred_vy,
            "speed_log_var": speed_log_var,
        })
        return out_df


def run_tcn_ekf_adapter(
    data: Union[TrajectoryData, pd.DataFrame],
    checkpoint_path: str = "model/best_model_real.pt",
    name: str = "TrackSense AI + EKF Pipeline",
) -> Union[PredictionData, pd.DataFrame]:
    """Helper function to execute TCN_EKF_ModelAdapter with default settings."""
    adapter = TCN_EKF_ModelAdapter(checkpoint_path=checkpoint_path, name=name)
    return adapter.predict(data)
