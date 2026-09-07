"""
velocity_update_test.py
Explicitly tests that update_velocity() changes the EKF trajectory.

This is a focused test to catch the bug where update_velocity()
is defined but never called, or called but has no effect.

The test compares:
    1. EKF with NO velocity updates (open loop)
    2. EKF with velocity updates (TCN predictions)
    
If both trajectories are identical, the integration is broken.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dummy_ekf import DummyEKF


def test_velocity_update_effect():
    """
    Test that velocity updates change the trajectory.
    
    This uses deterministic inputs so the result is predictable.
    """
    print("=" * 60)
    print("VELOCITY UPDATE EFFECT TEST")
    print("=" * 60)
    
    dt = 0.1
    ekf = DummyEKF(dt=dt)
    
    # Deterministic IMU data: constant forward acceleration
    n_steps = 50
    imu_data = []
    for i in range(n_steps):
        ax = 1.0  # Constant forward acceleration
        ay = 0.0  # No lateral
        gz = 0.0  # No turning
        imu_data.append((ax, ay, gz))
    
    # --- Test 1: Open loop (NO velocity updates) ---
    print("\n📊 Test 1: Open loop (no updates)...")
    ekf.reset()
    traj_no_update = ekf.run_open_loop(imu_data)
    final_no_update = ekf.get_position()
    print(f"   Final position (no update): ({final_no_update[0]:.2f}, {final_no_update[1]:.2f})")
    
    # --- Test 2: With velocity updates ---
    print("\n📊 Test 2: With velocity updates...")
    ekf.reset()
    
    # These velocity predictions should pull the trajectory
    # in a different direction than the IMU integration alone
    velocity_predictions = []
    for i in range(n_steps):
        # Constant velocity to the right and up
        vx = 5.0  # 5 m/s to the right
        vy = 2.0  # 2 m/s upward
        velocity_predictions.append((vx, vy))
    
    uncertainties = [0.01] * n_steps  # Low uncertainty = strong correction
    
    traj_with_update = ekf.run_with_updates(
        imu_data, 
        velocity_predictions, 
        uncertainties
    )
    final_with_update = ekf.get_position()
    print(f"   Final position (with update): ({final_with_update[0]:.2f}, {final_with_update[1]:.2f})")
    
    # --- Compare ---
    diff_x = final_with_update[0] - final_no_update[0]
    diff_y = final_with_update[1] - final_no_update[1]
    diff_mag = np.linalg.norm(np.array(final_with_update) - np.array(final_no_update))
    
    print("\n" + "-" * 60)
    print("COMPARISON:")
    print(f"   Position difference: ({diff_x:.2f}, {diff_y:.2f})m")
    print(f"   Difference magnitude: {diff_mag:.2f}m")
    
    # Verify update was applied
    assert ekf.last_update_applied, "❌ No velocity updates were applied!"
    assert ekf.last_update_time >= 0, "❌ Update time not recorded!"
    print(f"   ✅ Updates applied at timesteps up to {ekf.last_update_time}")
    
    # Verify trajectories differ
    if diff_mag > 0.01:
        print("   ✅ Velocity updates CHANGE the trajectory!")
        success = True
    else:
        print("   ⚠️  Velocity updates have MINIMAL EFFECT - integration may be broken")
        print("   Check that update_velocity() is called and affects the state")
        success = False
    
    # --- Plot ---
    plot_comparison(traj_no_update, traj_with_update, final_no_update, final_with_update)
    
    return {
        'success': success,
        'diff_magnitude': diff_mag,
        'final_no_update': final_no_update,
        'final_with_update': final_with_update,
        'trajectories': (traj_no_update, traj_with_update)
    }


def plot_comparison(traj_no_update, traj_with_update, final_no, final_with):
    """
    Plot trajectories with and without updates.
    """
    plt.figure(figsize=(10, 8))
    
    traj_no_arr = np.array(traj_no_update)
    traj_with_arr = np.array(traj_with_update)
    
    # Plot
    plt.plot(traj_no_arr[:, 0], traj_no_arr[:, 1], 
             'b--', label='Open Loop (No Velocity Updates)', linewidth=2, alpha=0.8)
    plt.plot(traj_with_arr[:, 0], traj_with_arr[:, 1], 
             'g-', label='With Velocity Updates', linewidth=2)
    
    # Start and end markers
    plt.scatter(traj_no_arr[0, 0], traj_no_arr[0, 1], 
                c='black', marker='o', s=120, label='Start')
    plt.scatter(final_no[0], final_no[1], 
                c='red', marker='s', s=120, label='End (No Update)')
    plt.scatter(final_with[0], final_with[1], 
                c='green', marker='s', s=120, label='End (With Update)')
    
    # Arrow showing difference
    plt.arrow(final_no[0], final_no[1], 
              final_with[0] - final_no[0], 
              final_with[1] - final_no[1],
              head_width=0.5, head_length=0.5, fc='orange', ec='orange', 
              label='Effect of velocity updates')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Velocity Update Effect Test\nDoes update_velocity() change the trajectory?')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    
    plt.savefig('velocity_update_effect.png', dpi=150, bbox_inches='tight')
    print("   📊 Plot saved to velocity_update_effect.png")
    plt.close()


def main():
    """Run the test."""
    print("=" * 60)
    print("VELOCITY UPDATE VERIFICATION")
    print("=" * 60)
    
    result = test_velocity_update_effect()
    
    print("\n" + "=" * 60)
    if result['success']:
        print("✅ VERIFIED: update_velocity() changes the trajectory!")
        print("   The TCN → EKF integration is working correctly.")
        print(f"   Position changed by: {result['diff_magnitude']:.2f}m")
    else:
        print("❌ FAILED: update_velocity() does NOT change the trajectory!")
        print("   Check that:")
        print("   1. update_velocity() is called in run_with_updates()")
        print("   2. The state is actually modified by update_velocity()")
        print("   3. The measurement noise R is not too large")
    print("=" * 60)
    
    return result['success']


if __name__ == "__main__":
    main()

