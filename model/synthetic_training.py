"""
synthetic_training.py
Training loop using synthetic data to test the entire pipeline.
Used in Hour 3-6 to verify training works before real data arrives.

COMPONENTS:
1. Gaussian Negative Log-Likelihood loss function
2. Training loop with validation
3. Learning rate scheduling
4. Model checkpointing
5. Training curve plotting
"""
import os
import torch
if hasattr(os, 'cpu_count') and os.cpu_count():
    torch.set_num_threads(os.cpu_count())
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from model import build_model

# Always save outputs next to this script, regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def gaussian_nll_loss(pred_mean, pred_log_var, target):
    """
    Gaussian Negative Log-Likelihood Loss with uncertainty.

    This loss function allows the model to predict both the mean and
    the uncertainty (log variance) of its prediction. The uncertainty
    is used by the EKF to weigh predictions appropriately.

    Args:
        pred_mean: Predicted mean (batch_size,)
        pred_log_var: Predicted log variance (batch_size,)
        target: Ground truth (batch_size,)

    Returns:
        Scalar loss value

    Math:
        Loss = 0.5 * exp(-log_var) * (target - mean)^2 + 0.5 * log_var
    """
    return torch.mean(
        0.5 * torch.exp(-pred_log_var) * (target - pred_mean) ** 2 +
        0.5 * pred_log_var
    )


def generate_training_data(n_samples=5000, window_size=100, val_split=0.2):
    """
    Generate synthetic data for training.

    Uses the same data generator as test_synthetic.py but with
    train/validation split.

    Returns:
        X_train, y_train, X_val, y_val
    """
    print("[..] Generating training data...")

    # Use the same generator
    from test_synthetic import generate_synthetic_data

    X, y = generate_synthetic_data(n_samples=n_samples, window_size=window_size)

    # Split
    split_idx = int(n_samples * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"   Train: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")

    return X_train, y_train, X_val, y_val


def train_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: TCN model
        dataloader: DataLoader with training data
        optimizer: Optimizer (AdamW)
        device: 'cuda' or 'cpu'

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    n_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        output = model(X_batch)
        mean, log_var, _ = output[:, 0], output[:, 1], output[:, 2]

        # Loss
        loss = gaussian_nll_loss(mean, log_var, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)

    return total_loss / n_samples


def validate_epoch(model, dataloader, device):
    """
    Validate for one epoch.

    Args:
        model: TCN model
        dataloader: DataLoader with validation data
        device: 'cuda' or 'cpu'

    Returns:
        Average loss for the validation set
    """
    model.eval()
    total_loss = 0
    n_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(X_batch)
            mean, log_var, _ = output[:, 0], output[:, 1], output[:, 2]
            loss = gaussian_nll_loss(mean, log_var, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            n_samples += X_batch.size(0)

    return total_loss / n_samples


def plot_training_curves(train_losses, val_losses, save_path='training_curves.png'):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TCN Training Curves (Synthetic Data)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Training curves saved to {save_path}")
    plt.close()


def main():
    """
    Main training loop with synthetic data.
    """
    print("=" * 60)
    print("SYNTHETIC DATA TRAINING")
    print("=" * 60)

    # Configuration
    WINDOW_SIZE = 100
    BATCH_SIZE = 256
    EPOCHS = 50
    LEARNING_RATE = 1e-3

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Generate data
    X_train, y_train, X_val, y_val = generate_training_data(
        n_samples=5000,
        window_size=WINDOW_SIZE
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[OK] Data prepared")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Build model
    model = build_model().to(device)
    print(f"[OK] Model built")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training loop
    print(f"\n>> Starting training for {EPOCHS} epochs...")
    print("-" * 60)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:2d}: train={train_loss:.4f}, val={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(SCRIPT_DIR, 'best_model_synthetic.pt')
            torch.save(model.state_dict(), save_path)
            print(f"  [OK] New best model saved!")

    print("-" * 60)
    print("[OK] Training complete!")

    # Plot curves
    plot_training_curves(train_losses, val_losses,
                         save_path=os.path.join(SCRIPT_DIR, 'training_curves.png'))

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Model saved to: {os.path.join(SCRIPT_DIR, 'best_model_synthetic.pt')}")

    return model, train_losses, val_losses


if __name__ == "__main__":
    main()
