"""
p2_model_integration.py
-----------------------
Person 2's TCN Model Wrapper - Integrates the trained TCN model into the EKF pipeline.
Model output: [velocity_mean, velocity_log_variance, heading_rate]
- velocity_mean: km/h
- velocity_log_variance: log((km/h)^2) 
- heading_rate: rad/s (yaw rate, NOT absolute heading)
"""

import numpy as np
import onnxruntime as ort
import torch

class P2ModelWrapper:
    def __init__(self, model_path, use_onnx=True, window_size=100):
        """
        Args:
            model_path: Path to .onnx or .pt file
            use_onnx: If True, load ONNX; else load PyTorch
            window_size: Expected input sequence length (100 = 10s at 10Hz)
        """
        self.window_size = window_size
        self.use_onnx = use_onnx
        self.input_shape = (1, 6, window_size)  # (batch, channels, seq_len)
        
        if use_onnx:
            self.session = ort.InferenceSession(model_path)
            # Get input name from model
            self.input_name = self.session.get_inputs()[0].name
        else:
            # Fallback to PyTorch
            from p2_tcn_model import TCNVelocityHeadingEstimator
            self.model = TCNVelocityHeadingEstimator(
                input_channels=6,
                output_channels=3,
                num_channels=[64, 128, 128, 64],
                kernel_size=3,
                dropout=0.25
            )
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
    
    def predict(self, imu_window):
        """
        Run inference on a single IMU window.
        
        Args:
            imu_window: numpy array of shape (6, window_size)
                      Channels order: [accel_x, accel_y, accel_z, 
                                       gyro_yaw, gyro_pitch, gyro_roll]
        
        Returns:
            velocity_mean: float (km/h)
            heading_rate: float (rad/s) 
            velocity_log_variance: float log((km/h)^2)
        """
        # Ensure correct shape
        if imu_window.shape != (6, self.window_size):
            raise ValueError(f"Expected shape (6, {self.window_size}), got {imu_window.shape}")
        
        # Add batch dimension -> (1, 6, window_size)
        input_tensor = imu_window.reshape(1, 6, self.window_size).astype(np.float32)
        
        if self.use_onnx:
            outputs = self.session.run(None, {self.input_name: input_tensor})
            # outputs[0] shape: (1, 3)
            result = outputs[0][0]
        else:
            with torch.no_grad():
                outputs = self.model(torch.from_numpy(input_tensor).float())
                result = outputs[0].numpy()
        
        velocity_mean = float(result[0])
        velocity_log_variance = float(result[1])
        heading_rate = float(result[2])
        
        return velocity_mean, heading_rate, velocity_log_variance
    
    def get_ekf_update(self, imu_window):
        """
        Get values ready for EKF update_velocity().
        
        Args:
            imu_window: numpy array of shape (6, window_size)
        
        Returns:
            speed_mps: float (m/s)
            heading_rate_radps: float (rad/s)
            velocity_variance_mps2: float ((m/s)^2)
            velocity_log_variance: float log((km/h)^2) - for EKF call signature
        """
        vel_kmh, heading_rate, log_var = self.predict(imu_window)
        
        # Convert speed to m/s
        speed_mps = vel_kmh / 3.6
        
        # Convert log variance from (km/h)^2 to (m/s)^2
        velocity_variance_mps2 = np.exp(log_var) / (3.6 ** 2)
        velocity_variance_mps2 = max(velocity_variance_mps2, 1e-3)  # Clamp for stability
        
        # Keep original log variance for EKF update_velocity signature
        # (it expects log((km/h)^2) internally)
        return speed_mps, heading_rate, velocity_variance_mps2, log_var
    
    def build_window(self, df, idx):
        """
        Extract a 100-step IMU window from dataframe at index idx.
        
        Args:
            df: pandas DataFrame with IMU columns
            idx: Current index (window ends at idx)
        
        Returns:
            numpy array of shape (6, window_size) or None if insufficient data
        """
        if idx < self.window_size:
            return None
        
        # Extract 6 channels in the exact order P2 expects
        window = df.iloc[idx - self.window_size:idx][
            ['accel_x', 'accel_y', 'accel_z', 
             'gyro_yaw', 'gyro_pitch', 'gyro_roll']
        ].to_numpy().T  # Transpose to (6, window_size)
        
        # Ensure float32 for ONNX
        return window.astype(np.float32)
