"""
train_simple.py
Minimal training loop with dummy EKF integration.

This is a simple training script that:
1. Trains the TCN on synthetic data (target speed in km/h)
2. Runs the trained model through the EKF (speed converted to m/s)
3. Compares trajectory quality before/after training

The goal is to show that the model learns to produce
velocity predictions that improve the EKF trajectory.
"""
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

import torch
if hasattr(os, 'cpu_count') and os.cpu_count():
    torch.set_num_threads(os.cpu_count())
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import build_model
from dummy_ekf import DummyEKF
from synthetic_training import gaussian_nll_loss, generate_training_data
from test_synthetic import generate_synthetic_data


def train_simple_model(epochs=30, batch_size=256, lr=1e-3):
    """
    Simple training loop on synthetic data.
    
    Target speed y is in km/h.
    
    Returns:
        model: Trained TCN model
        losses: Training loss history
    """
    print("=" * 60)
    print("SIMPLE TRAINING WITH EKF INTEGRATION (SPEED IN KM/H)")
    print("=" * 60)
    
    # Generate training data (y in km/h)
    print("\n1️⃣ Generating training data (speed in km/h)...")
    X_train, y_train, X_val, y_val = generate_training_data(
        n_samples=2000,
        window_size=100,
        val_split=0.2
    )
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    
    # Create dataloaders
    from torch.utils.data import DataLoader, TensorDataset
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    print("\n2️⃣ Building model...")
    model = build_model()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print("\n3️⃣ Training...")
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            output = model(X_batch)
            mean, log_var, _ = output[:, 0], output[:, 1], output[:, 2]
            loss = gaussian_nll_loss(mean, log_var, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                output = model(X_batch)
                mean, log_var, _ = output[:, 0], output[:, 1], output[:, 2]
                loss = gaussian_nll_loss(mean, log_var, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        
        val_loss = total_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch:2d}: train={train_loss:.4f}, val={val_loss:.4f}")
    
    print("✅ Training complete!")
    
    return model, train_losses, val_losses


def evaluate_with_ekf(model, n_samples=100):
    """
    Evaluate the trained model with the EKF.
    
    Converts TCN speed output (km/h) to m/s for EKF update:
    speed_mps = speed_kmh / 3.6
    
    Compares:
    1. EKF with untrained model (random)
    2. EKF with trained model
    """
    print("\n4️⃣ Evaluating with EKF...")
    
    # Generate test data
    X_test, y_test = generate_synthetic_data(n_samples=n_samples, window_size=100)
    
    # IMU data from X_test
    imu_data = []
    for i in range(n_samples):
        ax = X_test[i, 0, -1]
        ay = X_test[i, 1, -1]
        gz = X_test[i, 3, -1]
        imu_data.append((ax, ay, gz))
    
    # --- Run with UNTRAINED model ---
    print("\n   Running with UNTRAINED model...")
    untrained_model = build_model()
    untrained_model.eval()
    
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    with torch.no_grad():
        output_untrained = untrained_model(X_test_t)
    
    speed_mean_kmh_untrained = output_untrained[:, 0].numpy()
    speed_log_var_untrained = output_untrained[:, 1].numpy()
    heading_rate_untrained = output_untrained[:, 2].numpy()
    
    ekf_untrained = DummyEKF(dt=0.1)
    velocity_predictions = []
    uncertainties = []
    heading = 0.0
    
    for i in range(min(len(speed_mean_kmh_untrained), len(imu_data))):
        speed_kmh = speed_mean_kmh_untrained[i]
        speed_mps = speed_kmh / 3.6
        heading = heading + (heading_rate_untrained[i] if i < len(heading_rate_untrained) else 0) * 0.1
        vx = speed_mps * np.cos(heading)
        vy = speed_mps * np.sin(heading)
        velocity_predictions.append((vx, vy))
        var_kmh2 = np.exp(speed_log_var_untrained[i]) if i < len(speed_log_var_untrained) else 0.01
        uncertainties.append(var_kmh2 / (3.6 ** 2))
    
    traj_untrained = ekf_untrained.run_with_updates(imu_data, velocity_predictions, uncertainties)
    final_untrained = ekf_untrained.get_position()
    print(f"   Final position (untrained): ({final_untrained[0]:.2f}, {final_untrained[1]:.2f})")
    
    # --- Run with TRAINED model ---
    print("\n   Running with TRAINED model...")
    model.eval()
    
    with torch.no_grad():
        output_trained = model(X_test_t)
    
    speed_mean_kmh_trained = output_trained[:, 0].numpy()
    speed_log_var_trained = output_trained[:, 1].numpy()
    heading_rate_trained = output_trained[:, 2].numpy()
    
    ekf_trained = DummyEKF(dt=0.1)
    velocity_predictions = []
    uncertainties = []
    heading = 0.0
    
    for i in range(min(len(speed_mean_kmh_trained), len(imu_data))):
        speed_kmh = speed_mean_kmh_trained[i]
        speed_mps = speed_kmh / 3.6
        heading = heading + (heading_rate_trained[i] if i < len(heading_rate_trained) else 0) * 0.1
        vx = speed_mps * np.cos(heading)
        vy = speed_mps * np.sin(heading)
        velocity_predictions.append((vx, vy))
        var_kmh2 = np.exp(speed_log_var_trained[i]) if i < len(speed_log_var_trained) else 0.01
        uncertainties.append(var_kmh2 / (3.6 ** 2))
    
    traj_trained = ekf_trained.run_with_updates(imu_data, velocity_predictions, uncertainties)
    final_trained = ekf_trained.get_position()
    print(f"   Final position (trained): ({final_trained[0]:.2f}, {final_trained[1]:.2f})")
    
    # --- Compare ---
    diff_untrained = np.linalg.norm(np.array(final_untrained))
    diff_trained = np.linalg.norm(np.array(final_trained))
    
    print("\n   Comparison:")
    print(f"   Distance from start (untrained): {diff_untrained:.2f}m")
    print(f"   Distance from start (trained): {diff_trained:.2f}m")
    print(f"   Improvement: {(diff_untrained - diff_trained):.2f}m")
    
    # Plot trajectories
    plot_trajectory_comparison(traj_untrained, traj_trained, final_untrained, final_trained)
    
    return {
        'traj_untrained': traj_untrained,
        'traj_trained': traj_trained,
        'final_untrained': final_untrained,
        'final_trained': final_trained,
        'improvement': diff_untrained - diff_trained
    }


def plot_trajectory_comparison(traj_untrained, traj_trained, final_untrained, final_trained):
    """Plot trajectories from untrained vs trained model."""
    plt.figure(figsize=(10, 8))
    
    traj_untrained_arr = np.array(traj_untrained)
    traj_trained_arr = np.array(traj_trained)
    
    plt.plot(traj_untrained_arr[:, 0], traj_untrained_arr[:, 1], 
             'r--', label='Untrained Model (Random)', linewidth=2, alpha=0.7)
    plt.plot(traj_trained_arr[:, 0], traj_trained_arr[:, 1], 
             'g-', label='Trained Model', linewidth=2)
    
    plt.scatter(traj_untrained_arr[0, 0], traj_untrained_arr[0, 1], 
                c='black', marker='o', s=120, label='Start')
    plt.scatter(final_untrained[0], final_untrained[1], 
                c='red', marker='s', s=120, label='End (Untrained)')
    plt.scatter(final_trained[0], final_trained[1], 
                c='green', marker='s', s=120, label='End (Trained)')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('TCN Training Effect on EKF Trajectory\n(Model outputs speed in km/h → EKF in m/s)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.axis('equal')
    
    plt.savefig('training_effect_ekf.png', dpi=150, bbox_inches='tight')
    print("   📊 Plot saved to training_effect_ekf.png")
    plt.close()


def main():
    """Run simple training with EKF integration test."""
    print("=" * 60)
    print("SIMPLE TRAINING + EKF INTEGRATION")
    print("=" * 60)
    
    # Train model
    model, train_losses, val_losses = train_simple_model(epochs=30)
    
    # Evaluate with EKF
    results = evaluate_with_ekf(model, n_samples=100)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"EKF improvement: {results['improvement']:.2f}m")
    
    if results['improvement'] > 0.01:
        print("✅ Training improves EKF trajectory!")
    else:
        print("⚠️  Training shows minimal improvement - check data quality")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

