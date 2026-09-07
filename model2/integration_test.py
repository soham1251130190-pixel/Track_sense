"""
integration_test.py
Tests the complete pipeline: TCN → DummyEKF.

Verifies:
1. TCN produces valid outputs (3 values per sample, speed in km/h)
2. DummyEKF accepts TCN outputs (converted to m/s)
3. Trajectory changes when TCN predictions are applied
4. No errors occur during integration

This is the precursor to Sync 1 - proving plumbing works before real data.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import build_model
from dummy_ekf import DummyEKF
from test_synthetic import generate_synthetic_data


def run_integration_test():
    """
    End-to-end integration test:
    1. Generate synthetic data
    2. Run TCN model (untrained - random weights)
    3. Feed TCN outputs into DummyEKF
    4. Verify trajectory changes
    5. Plot results
    """
    print("=" * 60)
    print("TCN → EKF INTEGRATION TEST")
    print("=" * 60)
    
    # 1. Generate data
    print("\n1️⃣ Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=100, window_size=100)
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Target speed range: {y.min():.2f} - {y.max():.2f} km/h")
    
    # 2. Run TCN (untrained - random weights)
    print("\n2️⃣ Running TCN (untrained)...")
    model = build_model()
    model.eval()
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    with torch.no_grad():
        output = model(X_tensor)
    
    speed_mean_kmh = output[:, 0].numpy()
    speed_log_var = output[:, 1].numpy()
    heading_rate = output[:, 2].numpy()
    
    print(f"   Output shape: {output.shape}")
    print(f"   Speed mean range (km/h): {speed_mean_kmh.min():.2f} - {speed_mean_kmh.max():.2f} km/h")
    print(f"   Log var range: {speed_log_var.min():.2f} - {speed_log_var.max():.2f}")
    print(f"   Heading rate range: {heading_rate.min():.2f} - {heading_rate.max():.2f} rad/s")
    
    # 3. Prepare IMU data for EKF
    print("\n3️⃣ Preparing IMU data for EKF...")
    # Use the first 100 samples of X for IMU data
    # X shape: (samples, 6, window_size)
    # We need: accel_x, accel_y, gyro_z (yaw)
    imu_data = []
    for i in range(100):  # Use 100 samples
        ax = X[i, 0, -1]  # accel_x, last timestep
        ay = X[i, 1, -1]  # accel_y, last timestep
        gz = X[i, 3, -1]  # gyro_yaw, last timestep
        imu_data.append((ax, ay, gz))
    
    print(f"   IMU data length: {len(imu_data)}")
    
    # 4. Run EKF with TCN predictions
    print("\n4️⃣ Running EKF with TCN predictions (converting km/h → m/s)...")
    ekf = DummyEKF(dt=0.1)
    
    # Convert speed predictions (km/h) to vx, vy (m/s) using heading
    # speed_mps = speed_kmh / 3.6
    heading = 0.0  # Start with zero heading
    velocity_predictions = []
    uncertainties = []
    
    for i in range(min(len(speed_mean_kmh), len(imu_data))):
        # TCN outputs speed in km/h -> convert to m/s for EKF
        speed_kmh = speed_mean_kmh[i]
        speed_mps = speed_kmh / 3.6
        heading_rate_val = heading_rate[i] if i < len(heading_rate) else 0.0
        
        # Update heading (simplified)
        heading = heading + heading_rate_val * 0.1
        
        # Project speed (m/s) into vx, vy
        vx = speed_mps * np.cos(heading)
        vy = speed_mps * np.sin(heading)
        velocity_predictions.append((vx, vy))
        
        # Uncertainty from log_var (scaled to (m/s)^2)
        var_kmh2 = np.exp(speed_log_var[i]) if i < len(speed_log_var) else 0.01
        var_mps2 = var_kmh2 / (3.6 ** 2)
        uncertainties.append(var_mps2)
    
    print(f"   Velocity predictions: {len(velocity_predictions)}")
    print(f"   Uncertainties: {len(uncertainties)}")
    
    # Run EKF with updates
    traj_with_updates = ekf.run_with_updates(
        imu_data, 
        velocity_predictions, 
        uncertainties
    )
    final_pos_with = ekf.get_position()
    print(f"   Final position (with TCN): ({final_pos_with[0]:.2f}, {final_pos_with[1]:.2f})")
    
    # 5. Run EKF without TCN (open loop)
    print("\n5️⃣ Running EKF without TCN (open loop)...")
    ekf.reset()
    traj_without = ekf.run_open_loop(imu_data)
    final_pos_without = ekf.get_position()
    print(f"   Final position (open loop): ({final_pos_without[0]:.2f}, {final_pos_without[1]:.2f})")
    
    # 6. Compare trajectories
    print("\n6️⃣ Comparing trajectories...")
    diff = np.array(final_pos_with) - np.array(final_pos_without)
    diff_mag = np.linalg.norm(diff)
    print(f"   Position difference: ({diff[0]:.2f}, {diff[1]:.2f})m")
    print(f"   Diff magnitude: {diff_mag:.2f}m")
    
    if diff_mag > 0.01:
        print("   ✅ TCN predictions CHANGE the EKF trajectory!")
    else:
        print("   ⚠️  TCN predictions have MINIMAL EFFECT - check integration")
    
    # 7. Plot results
    print("\n7️⃣ Plotting trajectories...")
    plot_trajectories(traj_without, traj_with_updates, final_pos_without, final_pos_with)
    
    return {
        'with_tcn': traj_with_updates,
        'without_tcn': traj_without,
        'diff_magnitude': diff_mag,
        'success': diff_mag > 0.01
    }


def plot_trajectories(traj_without, traj_with, pos_without, pos_with):
    """
    Plot trajectories with and without TCN predictions.
    """
    plt.figure(figsize=(10, 8))
    
    # Convert trajectories to arrays
    traj_without_arr = np.array(traj_without)
    traj_with_arr = np.array(traj_with)
    
    # Plot trajectories
    plt.plot(traj_without_arr[:, 0], traj_without_arr[:, 1], 
             'b--', label='Open Loop (No TCN)', linewidth=2, alpha=0.7)
    plt.plot(traj_with_arr[:, 0], traj_with_arr[:, 1], 
             'g-', label='With TCN Predictions', linewidth=2)
    
    # Mark start and end
    plt.scatter(traj_without_arr[0, 0], traj_without_arr[0, 1], 
                c='black', marker='o', s=100, label='Start')
    plt.scatter(pos_without[0], pos_without[1], 
                c='red', marker='s', s=100, label='End (No TCN)')
    plt.scatter(pos_with[0], pos_with[1], 
                c='green', marker='s', s=100, label='End (With TCN)')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('TCN → EKF Integration Test\nTrajectory Comparison (TCN Speed in km/h → EKF in m/s)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    
    plt.savefig('integration_test_plot.png', dpi=150, bbox_inches='tight')
    print("   📊 Plot saved to integration_test_plot.png")
    plt.close()


def verify_update_velocity_is_called():
    """
    Verify that update_velocity() is actually called.
    
    This is a critical test - earlier versions of the code
    had a bug where update_velocity() was defined but never called.
    """
    print("\n" + "=" * 60)
    print("VERIFY update_velocity() IS CALLED")
    print("=" * 60)
    
    ekf = DummyEKF(dt=0.1)
    
    # Reset and check initial state
    ekf.reset()
    assert ekf.last_update_applied == False, "Last update should be False initially"
    assert ekf.last_update_time == -1, "Last update time should be -1 initially"
    print("✅ Initial state correct (no updates applied)")
    
    # Call update_velocity directly
    ekf.update_velocity(1.0, 0.5, 0.01)
    
    # Check that update was recorded
    assert ekf.last_update_applied == True, "Last update should be True after update"
    assert ekf.last_update_time >= 0, "Last update time should be set"
    print(f"✅ update_velocity() called! Time: {ekf.last_update_time}")
    
    # Test with the run_with_updates method
    ekf.reset()
    imu_data = [(0.0, 0.0, 0.0)] * 10
    velocity_predictions = [(1.0, 0.0)] * 10
    uncertainties = [0.01] * 10
    
    ekf.run_with_updates(imu_data, velocity_predictions, uncertainties)
    
    assert ekf.last_update_applied == True, "Updates should be applied in run_with_updates"
    assert ekf.last_update_time >= 0, "Update time should be recorded"
    print(f"✅ update_velocity() called {ekf.last_update_time + 1} times in run_with_updates")
    
    return True


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Verify update_velocity() is called", verify_update_velocity_is_called),
        ("Full TCN → EKF integration", run_integration_test),
    ]
    
    passed = 0
    results = {}
    
    for name, test_fn in tests:
        print(f"\n📌 Running: {name}")
        try:
            result = test_fn()
            if isinstance(result, dict):
                success = result.get('success', False)
            else:
                success = result if isinstance(result, bool) else True
            
            if success:
                print(f"   ✅ {name} PASSED")
                passed += 1
            else:
                print(f"   ⚠️  {name} COMPLETED with warnings")
                passed += 0.5
        except Exception as e:
            print(f"   ❌ {name} FAILED: {e}")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{len(tests)} tests passed")
    print("=" * 60)
    
    if passed == len(tests):
        print("🎉 All integration tests passed!")
        print("   The TCN → EKF pipeline is ready for real data!")
    else:
        print("⚠️  Some tests failed. Review the output and fix issues.")
    
    return passed == len(tests)


if __name__ == "__main__":
    main()
    
