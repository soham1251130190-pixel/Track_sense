"""
dummy_ekf.py
Simplified EKF stub for integration testing between TCN and EKF.

This is NOT the production EKF - that's Person 3's responsibility.
This is a minimal implementation that:
1. Accepts the TCN's velocity predictions
2. Verifies the update_velocity() path works
3. Shows that the trajectory changes when predictions are applied

State: [x, y, vx, vy, heading, gyro_bias]
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np


class DummyEKF:
    """
    Minimal EKF for testing TCN integration.
    
    State vector (6):
        x, y: position in local frame (m)
        vx, vy: velocity in local frame (m/s)
        heading: vehicle heading (rad)
        gyro_bias: gyroscope bias (rad/s)
    
    This stub is only for verifying that:
        1. update_velocity() is called correctly
        2. The trajectory changes when predictions are applied
        3. The TCN output format matches the EKF input format
    """
    
    def __init__(self, dt=0.1):
        self.dt = dt
        self.x = np.zeros(6)  # [x, y, vx, vy, heading, gyro_bias]
        self.P = np.eye(6) * 1.0
        self.trajectory = []  # Store trajectory for testing
        self.last_update_applied = False
        self.last_update_time = -1
        
        # Process noise
        self.Q = np.diag([0.05, 0.05, 0.5, 0.5, 0.01, 1e-4])
        
        # Measurement noise for velocity update (will be set by TCN)
        self.R_velocity = np.eye(2) * 0.1
        
        print("✅ DummyEKF initialized")
        print(f"   State: {self.x}")
        print(f"   dt: {self.dt}s")
    
    def predict(self, accel_x, accel_y, gyro_z):
        """
        Prediction step using IMU measurements.
        
        This is a simplified kinematic model:
            heading = heading + (gyro_z - gyro_bias) * dt
            vx = vx + accel_x * cos(heading) * dt
            vy = vy + accel_y * sin(heading) * dt
            x = x + vx * dt
            y = y + vy * dt
        
        Args:
            accel_x: Accelerometer X (m/s²)
            accel_y: Accelerometer Y (m/s²)
            gyro_z: Gyroscope Z (rad/s) - yaw rate
        """
        dt = self.dt
        x, y, vx, vy, heading, bg = self.x
        
        # Update heading
        heading = heading + (gyro_z - bg) * dt
        
        # Rotate acceleration into world frame
        wx = accel_x * np.cos(heading) - accel_y * np.sin(heading)
        wy = accel_x * np.sin(heading) + accel_y * np.cos(heading)
        
        # Update velocity
        vx = vx + wx * dt
        vy = vy + wy * dt
        
        # Update position
        x = x + vx * dt
        y = y + vy * dt
        
        # Store new state
        self.x = np.array([x, y, vx, vy, heading, bg])
        self.trajectory.append((x, y))
    
    def update_velocity(self, vx_meas, vy_meas, R_scalar=0.1):
        """
        Update step using TCN's velocity prediction.
        
        THIS IS THE CRITICAL METHOD FOR INTEGRATION.
        It must be called correctly for the TCN to affect the EKF.
        
        Args:
            vx_meas: Measured velocity X (m/s) from TCN
            vy_meas: Measured velocity Y (m/s) from TCN
            R_scalar: Uncertainty from TCN (log_var → variance)
        
        Returns:
            bool: True if the update was applied
        """
        # H matrix: maps state to measurement (vx, vy)
        H = np.zeros((2, 6))
        H[0, 2] = 1  # vx
        H[1, 3] = 1  # vy
        
        # Measurement
        z = np.array([vx_meas, vy_meas])
        
        # Measurement noise (from TCN uncertainty)
        R = np.eye(2) * max(R_scalar, 1e-3)
        
        # Innovation
        y_resid = z - H @ self.x
        
        # Innovation covariance
        S = H @ self.P @ H.T + R
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # State correction
        self.x = self.x + K @ y_resid
        
        # Covariance correction
        self.P = (np.eye(6) - K @ H) @ self.P
        
        # Record that an update happened
        self.last_update_applied = True
        self.last_update_time = len(self.trajectory)
        
        return True
    
    def get_position(self):
        """Return current position (x, y)."""
        return self.x[0], self.x[1]
    
    def get_velocity(self):
        """Return current velocity (vx, vy)."""
        return self.x[2], self.x[3]
    
    def get_heading(self):
        """Return current heading (rad)."""
        return self.x[4]
    
    def reset(self):
        """Reset the filter state."""
        self.x = np.zeros(6)
        self.trajectory = []
        self.last_update_applied = False
        self.last_update_time = -1
    
    def run_open_loop(self, imu_data, dt=0.1):
        """
        Run open-loop predictions (no velocity updates).
        
        Args:
            imu_data: List of (accel_x, accel_y, gyro_z) tuples
            dt: Timestep
        
        Returns:
            List of (x, y) positions
        """
        self.dt = dt
        self.reset()
        
        for ax, ay, gz in imu_data:
            self.predict(ax, ay, gz)
        
        return self.trajectory
    
    def run_with_updates(self, imu_data, velocity_predictions, uncertainty_scalars, dt=0.1):
        """
        Run with velocity updates from TCN.
        
        Args:
            imu_data: List of (accel_x, accel_y, gyro_z) tuples
            velocity_predictions: List of (vx, vy) from TCN
            uncertainty_scalars: List of uncertainty values from TCN
            dt: Timestep
        
        Returns:
            List of (x, y) positions
        """
        self.dt = dt
        self.reset()
        
        for i, (ax, ay, gz) in enumerate(imu_data):
            # Predict
            self.predict(ax, ay, gz)
            
            # Update if we have a prediction for this timestep
            if i < len(velocity_predictions):
                vx, vy = velocity_predictions[i]
                R = uncertainty_scalars[i] if i < len(uncertainty_scalars) else 0.1
                self.update_velocity(vx, vy, R)
        
        return self.trajectory


def test_dummy_ekf():
    """Quick test to verify DummyEKF works."""
    print("=" * 60)
    print("DUMMY EKF TEST")
    print("=" * 60)
    
    ekf = DummyEKF(dt=0.1)
    
    # Generate synthetic IMU data
    n_steps = 100
    imu_data = []
    for i in range(n_steps):
        ax = 0.5 * np.sin(i * 0.05)  # Gentle acceleration
        ay = 0.1 * np.sin(i * 0.03)  # Slight lateral
        gz = 0.02 * np.sin(i * 0.02)  # Gentle turning
        imu_data.append((ax, ay, gz))
    
    # Run open loop
    print("\n📊 Running open loop...")
    traj_open = ekf.run_open_loop(imu_data)
    print(f"   Open loop trajectory length: {len(traj_open)}")
    print(f"   Final position: ({traj_open[-1][0]:.2f}, {traj_open[-1][1]:.2f})")
    
    # Run with velocity updates
    print("\n📊 Running with velocity updates...")
    velocity_predictions = [(1.0, 0.5)] * n_steps  # Constant velocity
    uncertainties = [0.01] * n_steps  # Low uncertainty
    traj_update = ekf.run_with_updates(imu_data, velocity_predictions, uncertainties)
    print(f"   Updated trajectory length: {len(traj_update)}")
    print(f"   Final position: ({traj_update[-1][0]:.2f}, {traj_update[-1][1]:.2f})")
    
    # Verify updates were applied
    assert ekf.last_update_applied, "No velocity updates applied!"
    assert ekf.last_update_time >= 0, "Update time not recorded!"
    
    # Verify trajectories differ
    diff = np.array(traj_update[-1]) - np.array(traj_open[-1])
    print(f"\n📊 Difference between trajectories:")
    print(f"   Position diff: ({diff[0]:.2f}, {diff[1]:.2f})m")
    print(f"   Diff magnitude: {np.linalg.norm(diff):.2f}m")
    
    if np.linalg.norm(diff) > 0.01:
        print("   ✅ update_velocity() changes the trajectory!")
    else:
        print("   ⚠️  update_velocity() has minimal effect - check implementation")
    
    return traj_open, traj_update


if __name__ == "__main__":
    test_dummy_ekf()

