"""
train_real.py
Training script for TCN on real IO-VNBD dataset (Member 1's data).

Architecture:
- Input: (batch, 6, 100) -> 6 IMU channels at 10Hz
- Target: speed_kmh (km/h)
- Output: (speed_mean, speed_log_var, heading_rate)
- Loss: Gaussian Negative Log-Likelihood (NLL) with uncertainty estimation
- Checkpoint: Saves best model to best_model_real.pt
"""
import os
import sys
import time
import json
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Optimize CPU threads for high throughput
try:
    threads = os.cpu_count() or 4
    torch.set_num_threads(threads)
except Exception:
    pass

from model import build_model, count_parameters
from real_data_loader import load_real_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def gaussian_nll_loss(pred_mean, pred_log_var, target):
    """
    Gaussian Negative Log-Likelihood Loss.
    Loss = 0.5 * exp(-log_var) * (target - mean)^2 + 0.5 * log_var
    """
    return torch.mean(
        0.5 * torch.exp(-pred_log_var) * (target - pred_mean) ** 2 +
        0.5 * pred_log_var
    )


def train_one_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        output = model(X_batch)
        mean, log_var, _ = output[:, 0], output[:, 1], output[:, 2]
        
        loss = gaussian_nll_loss(mean, log_var, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        batch_size = X_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
    return total_loss / total_samples


def evaluate_metrics(model, dataloader, device):
    """Evaluate loss, RMSE, and MAE on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    all_preds = []
    all_targets = []
    all_vars = []
    
    with torch.inference_mode():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            output = model(X_batch)
            mean, log_var, _ = output[:, 0], output[:, 1], output[:, 2]
            
            loss = gaussian_nll_loss(mean, log_var, y_batch)
            
            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            all_preds.extend(mean.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            all_vars.extend(torch.exp(log_var).cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_vars = np.array(all_vars)
    
    avg_loss = total_loss / total_samples
    rmse_kmh = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae_kmh = np.mean(np.abs(all_preds - all_targets))
    rmse_mps = rmse_kmh / 3.6
    mae_mps = mae_kmh / 3.6
    mean_uncertainty = np.mean(np.sqrt(all_vars))
    
    return {
        'loss': avg_loss,
        'rmse_kmh': float(rmse_kmh),
        'mae_kmh': float(mae_kmh),
        'rmse_mps': float(rmse_mps),
        'mae_mps': float(mae_mps),
        'mean_uncertainty_kmh': float(mean_uncertainty),
        'predictions': all_preds,
        'targets': all_targets
    }


def plot_real_training_curves(train_losses, val_losses, val_rmses, save_path):
    """Plot comprehensive loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train NLL Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r--', label='Val NLL Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Gaussian NLL Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # RMSE plot
    ax2.plot(epochs, val_rmses, 'g-', label='Validation Speed RMSE (km/h)', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE (km/h)')
    ax2.set_title('Validation Speed Prediction RMSE')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Training curves saved to: {save_path}")


def train_real_tcn(csv_path=None, epochs=50, batch_size=512, lr=1e-3, step_size=2, device_str=None):
    """
    Main training execution function.
    """
    print("=" * 60)
    print("TRAINING TCN ON REAL IO-VNBD DATA (HOURS 6-10)")
    print("=" * 60)
    
    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"🖥️  Using Compute Device: {device} ({torch.get_num_threads()} CPU threads)")
    
    # 1. Load Real Data with high temporal resolution (stride=2 for fast convergence)
    train_loader, val_loader, test_loader, meta = load_real_data(
        csv_path=csv_path,
        window_size=100,
        val_split=0.15,
        test_split=0.15,
        batch_size=batch_size,
        step_size=step_size
    )
    
    # 2. Build Model
    print("\n🧠 Instantiating TCN Model...")
    model = build_model().to(device)
    total_params = count_parameters(model)
    
    # 3. Setup Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float('inf')
    best_val_rmse = float('inf')
    best_epoch = -1
    
    best_model_path = os.path.join(SCRIPT_DIR, 'best_model_real.pt')
    latest_model_path = os.path.join(SCRIPT_DIR, 'latest_model_real.pt')
    
    train_losses = []
    val_losses = []
    val_rmses = []
    
    print(f"\n🚀 Starting Training ({epochs} epochs, lr={lr}, batch_size={batch_size})...")
    print("-" * 75)
    print(f"{'Epoch':^7} | {'Train NLL':^11} | {'Val NLL':^11} | {'Val RMSE(km/h)':^15} | {'Val MAE(km/h)':^15} | {'Time':^7}")
    print("-" * 75)
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_metrics(model, val_loader, device)
        
        scheduler.step()
        epoch_time = time.time() - t0
        
        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        val_rmses.append(val_metrics['rmse_kmh'])
        
        # Save checkpoint on improved validation loss
        is_best = False
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_rmse = val_metrics['rmse_kmh']
            best_epoch = epoch
            is_best = True
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_rmse_kmh': best_val_rmse,
                'val_mae_kmh': val_metrics['mae_kmh'],
                'total_params': total_params
            }, best_model_path)
            
        marker = " 🌟 [BEST]" if is_best else ""
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or is_best:
            print(f"{epoch:^7d} | {train_loss:^11.4f} | {val_metrics['loss']:^11.4f} | {val_metrics['rmse_kmh']:^15.2f} | {val_metrics['mae_kmh']:^15.2f} | {epoch_time:^6.1f}s{marker}")
            
    total_training_time = time.time() - start_time
    print("-" * 75)
    print(f"✅ Training completed in {total_training_time:.1f}s ({total_training_time/60:.2f} min)")
    print(f"🏆 Best Model saved at Epoch {best_epoch} with Val Loss = {best_val_loss:.4f}, Val RMSE = {best_val_rmse:.2f} km/h")
    
    # Save latest model
    torch.save(model.state_dict(), latest_model_path)
    
    # Plot curves
    curve_path = os.path.join(SCRIPT_DIR, 'real_training_curves.png')
    plot_real_training_curves(train_losses, val_losses, val_rmses, curve_path)
    
    # 4. Final Evaluation on Hold-Out Test Set using Best Checkpoint
    print("\n🎯 Evaluating Best Model on Hold-Out Test Set (15% Real Data)...")
    checkpoint = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate_metrics(model, test_loader, device)
    
    print(f"\n📊 FINAL TEST SET METRICS:")
    print(f"   Test NLL Loss:         {test_metrics['loss']:.4f}")
    print(f"   Test Speed RMSE:       {test_metrics['rmse_kmh']:.2f} km/h ({test_metrics['rmse_mps']:.2f} m/s)")
    print(f"   Test Speed MAE:        {test_metrics['mae_kmh']:.2f} km/h ({test_metrics['mae_mps']:.2f} m/s)")
    print(f"   Mean Predicted StdDev: ±{test_metrics['mean_uncertainty_kmh']:.2f} km/h")
    
    # Save training report metrics JSON
    metrics_summary = {
        'epochs': epochs,
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'best_val_rmse_kmh': float(best_val_rmse),
        'test_loss': float(test_metrics['loss']),
        'test_rmse_kmh': float(test_metrics['rmse_kmh']),
        'test_mae_kmh': float(test_metrics['mae_kmh']),
        'test_rmse_mps': float(test_metrics['rmse_mps']),
        'test_mae_mps': float(test_metrics['mae_mps']),
        'training_time_sec': float(total_training_time),
        'total_samples': meta['total_samples'],
        'train_samples': meta['train_samples'],
        'val_samples': meta['val_samples'],
        'test_samples': meta['test_samples']
    }
    
    metrics_json_path = os.path.join(SCRIPT_DIR, 'training_metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"📝 Training metrics saved to: {metrics_json_path}")
    
    return model, metrics_summary


if __name__ == "__main__":
    train_real_tcn(epochs=50, batch_size=512, lr=1e-3, step_size=2)
    
