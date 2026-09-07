"""
src/tracksense/eval/adapters.py
-------------------------------
Model adapter bridging VelocityTCN (Person 2) and VehicleEKF (Person 3)
for end-to-end evaluation in Person 4's Evaluation Harness.

Fixes all integration issues:
1. Converts heading_rate (rad/s) from TCN into accumulated absolute heading angle (rad).
2. Proper unit management (speed in km/h to EKF which converts internally).
3. Dynamic uncertainty weighting using TCN predicted log_variance.
4. Vectorized batch sliding-window inference with stride_tricks.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from tracksense.eval.datatypes import TrajectoryData, PredictionData

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "model2") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "model2"))
if str(ROOT_DIR / "model") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "model"))

try:
    from ekf_fusion import VehicleEKF
except ImportError:
    from model.dummy_ekf import DummyEKF as VehicleEKF

try:
    from model import build_model
except ImportError:
    from model2.model import build_model


class TCN_EKF_ModelAdapter:
    """
    Adapter running TrackSense (VelocityTCN + VehicleEKF) on TrajectoryData.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        window_size: int = 100,
        device: str = "cpu",
    ):
        self.window_size = window_size
        self.device = torch.device(device)
        self.onnx_session = None
        self.torch_model = None

        if checkpoint_path is None and onnx_path is None:
            for candidate in [
                ROOT_DIR / "model2" / "best_model_real.pt",
                ROOT_DIR / "model" / "best_model_real.pt",
                ROOT_DIR / "best_model_real.pt",
            ]:
                if candidate.exists():
                    checkpoint_path = str(candidate)
                    break

        if onnx_path and os.path.exists(onnx_path):
            import onnxruntime as ort
            self.onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            print(f"📦 Loaded ONNX model from: {onnx_path}")
        elif checkpoint_path and os.path.exists(checkpoint_path):
            self.torch_model = build_model().to(self.device)
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.torch_model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.torch_model.load_state_dict(ckpt)
            self.torch_model.eval()
            print(f"📦 Loaded PyTorch weights from: {checkpoint_path}")
        else:
            print("⚠️ No checkpoint found! Initializing untrained TCN.")
            self.torch_model = build_model().to(self.device)
            self.torch_model.eval()

    def _run_batched_tcn_inference(self, imu_6ch: np.ndarray, batch_size: int = 512) -> np.ndarray:
        N = len(imu_6ch)
        if N < self.window_size:
            pad_len = self.window_size - N
            imu_6ch = np.pad(imu_6ch, ((pad_len, 0), (0, 0)), mode="edge")
            N = len(imu_6ch)

        windows = np.lib.stride_tricks.sliding_window_view(
            imu_6ch, window_shape=(self.window_size, 6)
        ).squeeze(axis=1)

        windows = np.ascontiguousarray(np.transpose(windows, (0, 2, 1)), dtype=np.float32)
        num_windows = len(windows)

        raw_preds = []

        if self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            for start_idx in range(0, num_windows, batch_size):
                batch = windows[start_idx : start_idx + batch_size]
                out = self.onnx_session.run(None, {input_name: batch})[0]
                raw_preds.append(out)
        else:
            with torch.no_grad():
                for start_idx in range(0, num_windows, batch_size):
                    batch = torch.from_numpy(windows[start_idx : start_idx + batch_size]).to(self.device)
                    out = self.torch_model(batch)
                    raw_preds.append(out.cpu().numpy())

        raw_preds = np.vstack(raw_preds)

        warmup_pad = np.repeat(raw_preds[0:1, :], self.window_size - 1, axis=0)
        full_preds = np.vstack([warmup_pad, raw_preds])

        return full_preds

    def run_pipeline(self, gt_data: TrajectoryData) -> PredictionData:
        N = len(gt_data.timestamps)
        timestamps = gt_data.timestamps
        dt_arr = np.diff(timestamps, prepend=timestamps[0])

        if gt_data.imu_accel is not None and gt_data.imu_gyro is not None:
            imu_6ch = np.hstack([gt_data.imu_accel, gt_data.imu_gyro])
        else:
            raise ValueError("TrajectoryData must contain imu_accel and imu_gyro")

        tcn_out = self._run_batched_tcn_inference(imu_6ch)
        speed_mean_kmh = np.clip(tcn_out[:, 0], 0.0, 150.0)
        speed_log_var = tcn_out[:, 1]
        heading_rate_rads = tcn_out[:, 2]

        is_blackout = np.zeros(N, dtype=bool)
        for interval in gt_data.blackout_intervals:
            mask = (timestamps >= interval.start_time) & (timestamps <= interval.end_time)
            is_blackout |= mask

        if gt_data.gnss_position is not None:
            is_blackout |= np.isnan(gt_data.gnss_position[:, 0])

        p0 = gt_data.gt_position[0, :2]
        if N > 1 and np.linalg.norm(gt_data.gt_position[1, :2] - p0) > 1e-4:
            init_heading = np.arctan2(
                gt_data.gt_position[1, 1] - p0[1],
                gt_data.gt_position[1, 0] - p0[0]
            )
        elif gt_data.gt_velocity is not None and np.linalg.norm(gt_data.gt_velocity[0, :2]) > 0.1:
            init_heading = np.arctan2(gt_data.gt_velocity[0, 1], gt_data.gt_velocity[0, 0])
        else:
            init_heading = 0.0

        init_speed_ms = speed_mean_kmh[0] / 3.6
        ekf = VehicleEKF(initial_state=[p0[0], p0[1], init_speed_ms, init_heading])

        pred_positions = np.zeros((N, 2))
        pred_velocities = np.zeros((N, 2))
        current_heading = init_heading

        for i in range(N):
            dt = dt_arr[i] if dt_arr[i] > 0 else 0.1

            current_heading = current_heading + heading_rate_rads[i] * dt
            current_heading = (current_heading + np.pi) % (2 * np.pi) - np.pi

            ekf.predict(dt)

            if not is_blackout[i] and gt_data.gnss_position is not None:
                gx = gt_data.gnss_position[i, 0]
                gy = gt_data.gnss_position[i, 1]
                if not (np.isnan(gx) or np.isnan(gy)):
                    ekf.update_gps(gx, gy)
                    if i > 5:
                        dx = ekf.x[0] - pred_positions[i - 5, 0]
                        dy = ekf.x[1] - pred_positions[i - 5, 1]
                        if np.hypot(dx, dy) > 0.5:
                            gps_heading = np.arctan2(dy, dx)
                            current_heading = 0.8 * current_heading + 0.2 * gps_heading

            ekf.update_velocity(
                v_kmh=float(speed_mean_kmh[i]),
                heading_meas=float(current_heading),
                velocity_log_variance=float(speed_log_var[i]),
            )

            pred_positions[i] = ekf.x[0:2]
            v_est = ekf.x[2]
            h_est = ekf.x[3]
            pred_velocities[i] = [v_est * np.cos(h_est), v_est * np.sin(h_est)]

        return PredictionData(
            name="TrackSense AI+EKF Pipeline",
            timestamps=timestamps,
            predicted_position=pred_positions,
            predicted_velocity=pred_velocities,
            predicted_log_variance=speed_log_var,
        )
