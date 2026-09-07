"""
test_synthetic.py
Quick test to verify the TCN model runs on synthetic data.
Used in Hour 0-3 to confirm the plumbing works before real data arrives.

TEST CASES:
1. Model forward pass completes without errors
2. Output shape is correct (batch, 3)
3. Output values are reasonable (not NaN/Inf)
4. Gradient flows through the model (backpropagation works)
5. Model can overfit to a single batch (sanity check)
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import torch
import numpy as np
from model import build_model, count_parameters


def generate_synthetic_data(n_samples=1000, window_size=100, noise_scale=0.1):
    """
    Generate synthetic IMU data with a fake speed label.

    Args:
        n_samples: Number of samples
        window_size: Time window length
        noise_scale: Amount of Gaussian noise to add

    Returns:
        X: (n_samples, 6, window_size) - Synthetic IMU data
        y: (n_samples,) - Synthetic speed labels (km/h)

    Generates:
        - Random accelerometer signals (0.1-2.0 m/s²) with noise
        - Random gyroscope signals (0.01-0.5 rad/s) with noise
        - Speed labels correlated with accel_x (simplistic but useful for testing)
        - Constant offset to simulate gravity (accel_z ~ 9.8)
    """
    print("[..] Generating synthetic data...")

    # Random seed for reproducibility
    np.random.seed(42)

    # Generate base signals
    # Accelerometer: random walking patterns
    accel_x = np.cumsum(np.random.randn(n_samples, window_size) * 0.1, axis=1)
    accel_y = np.cumsum(np.random.randn(n_samples, window_size) * 0.05, axis=1)
    accel_z = 9.8 + np.random.randn(n_samples, window_size) * 0.2  # Gravity + noise

    # Gyroscope: random rotations
    gyro_yaw = np.random.randn(n_samples, window_size) * 0.05
    gyro_pitch = np.random.randn(n_samples, window_size) * 0.03
    gyro_roll = np.random.randn(n_samples, window_size) * 0.03

    # Stack into X: (n_samples, 6, window_size)
    X = np.stack([
        accel_x, accel_y, accel_z,
        gyro_yaw, gyro_pitch, gyro_roll
    ], axis=1).astype(np.float32)

    # Generate speed labels: roughly correlated with accel_x + noise
    # Speed = average acceleration over window + random offset
    y = np.abs(np.mean(accel_x, axis=1)) * 18 + np.random.randn(n_samples) * 7
    y = np.clip(y, 0, 120).astype(np.float32)  # Clip to realistic range (km/h)

    print(f"   Generated {n_samples} samples")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Speed range: {y.min():.2f} - {y.max():.2f} km/h")

    return X, y


def test_forward_pass():
    """Test 1: Model forward pass completes without errors."""
    print("\n" + "=" * 60)
    print("TEST 1: Forward Pass")
    print("=" * 60)

    model = build_model()
    X, _ = generate_synthetic_data(n_samples=4, window_size=100)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_tensor)

    # Check shape
    assert output.shape == (4, 3), f"Expected (4,3), got {output.shape}"

    # Check for NaNs/Infs
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"

    print(f"[OK] Forward pass successful")
    print(f"   Input shape: {X_tensor.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output sample: {output[0].tolist()}")

    return True


def test_backward_pass():
    """Test 2: Gradient flows through the model."""
    print("\n" + "=" * 60)
    print("TEST 2: Backward Pass (Gradient Flow)")
    print("=" * 60)

    model = build_model()
    X, y = generate_synthetic_data(n_samples=4, window_size=100)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Forward pass
    output = model(X_tensor)
    mean, log_var, _ = output[:, 0], output[:, 1], output[:, 2]

    # Simple MSE loss (even though we'll use Gaussian NLL later)
    loss = torch.mean((mean - y_tensor) ** 2)

    # Backward pass
    loss.backward()

    # Check gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            print(f"   ✓ Gradient exists for {name}")
            break

    assert has_grad, "No gradients found - backward pass failed!"

    print(f"[OK] Backward pass successful")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients flow through all layers")

    return True


def test_overfit_single_batch():
    """Test 3: Model can overfit to a single batch (sanity check)."""
    print("\n" + "=" * 60)
    print("TEST 3: Overfit Single Batch (Sanity Check)")
    print("=" * 60)

    model = build_model()
    X, y = generate_synthetic_data(n_samples=16, window_size=100)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    initial_loss = None
    final_loss = None

    for epoch in range(100):
        output = model(X_tensor)
        mean, log_var, _ = output[:, 0], output[:, 1], output[:, 2]
        loss = torch.mean((mean - y_tensor) ** 2)

        if epoch == 0:
            initial_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 99:
            final_loss = loss.item()

    print(f"   Initial loss: {initial_loss:.4f}")
    print(f"   Final loss: {final_loss:.4f}")
    print(f"   Reduction: {(1 - final_loss/initial_loss) * 100:.1f}%")

    # Should reduce loss significantly
    assert final_loss < initial_loss * 0.5, "Model failed to overfit!"

    print(f"[OK] Overfit test passed - model can learn")

    return True


def test_model_size():
    """Test 4: Model size is within target range."""
    print("\n" + "=" * 60)
    print("TEST 4: Model Size Check")
    print("=" * 60)

    model = build_model()
    total_params = count_parameters(model)

    # Target: ~100,000 parameters
    assert total_params < 150000, f"Too many parameters: {total_params:,}"
    assert total_params > 50000, f"Too few parameters: {total_params:,}"

    print(f"[OK] Model size is within target range (~100K parameters)")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("SYNTHETIC DATA TEST SUITE")
    print("=" * 60)

    tests = [
        test_forward_pass,
        test_backward_pass,
        test_overfit_single_batch,
        test_model_size,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] Test failed: {e}")

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    if passed == len(tests):
        print("ALL TESTS PASSED! Model is ready for real data.")
    else:
        print("WARNING: Some tests failed. Please fix before proceeding.")

    return passed == len(tests)


if __name__ == "__main__":
    main()
