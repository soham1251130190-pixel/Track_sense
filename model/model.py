"""
model.py
Temporal Convolutional Network (TCN) for IMU-based velocity estimation.

ARCHITECTURE SPECIFICATION:
- 5 TCN residual blocks with dilation: 1, 2, 4, 8, 16
- 64 channels per layer
- Kernel size: 3
- Dropout: 0.1
- Global average pooling
- Output head: 64 -> 32 -> 3 (speed_mean, speed_log_var, heading_rate)

INPUT:
    Shape: (batch_size, 6, window_size)
    - 6 = [accel_x, accel_y, accel_z, gyro_yaw, gyro_pitch, gyro_roll]
    - window_size = 100 (10 seconds at 10Hz)

OUTPUT:
    Shape: (batch_size, 3)
    - [speed_mean, speed_log_var, heading_rate]
    - speed_mean: predicted forward velocity (km/h)
    - speed_log_var: uncertainty (log variance) for EKF
    - heading_rate: rate of heading change (rad/s)

TOTAL PARAMETERS: ~100,000 (target)
"""
import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """
    Residual block with dilated causal convolutions.

    Architecture:
        Input -> Conv1D (dilated, causal) -> ReLU -> Dropout
              -> Conv1D (dilated, causal) -> ReLU -> Dropout
              -> + Residual -> ReLU -> Output

    Dilation schedule: 1, 2, 4, 8, 16 (exponential growth)

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        dilation: Dilation factor (2 ** layer_index)
        kernel_size: Kernel size (fixed at 3)
        dropout: Dropout rate (0.1)
    """
    def __init__(self, in_ch, out_ch, dilation, kernel_size=3, dropout=0.1):
        super().__init__()

        # Calculate causal padding: (kernel_size - 1) * dilation
        pad = (kernel_size - 1) * dilation

        # First dilated convolutional layer
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=pad, dilation=dilation
        )

        # Second dilated convolutional layer
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size,
            padding=pad, dilation=dilation
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection: 1x1 conv if channel dimensions change
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        # Remove causal padding from each conv output
        pad = self.conv1.padding[0]

        # First conv block
        out = self.relu(self.conv1(x)[:, :, :-pad])
        out = self.dropout(out)

        # Second conv block
        pad2 = self.conv2.padding[0]
        out = self.relu(self.conv2(out)[:, :, :-pad2])
        out = self.dropout(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)

        # Add residual and apply activation
        return self.relu(out + res)


class VelocityTCN(nn.Module):
    """
    TCN that estimates vehicle velocity from IMU signals.

    Architecture:
        5 TCN blocks with increasing dilation: 1, 2, 4, 8, 16
        64 channels per layer
        Global average pooling
        2-layer MLP head: 64 -> 32 -> 3

    Output:
        - speed_mean: predicted forward velocity (km/h)
        - speed_log_var: uncertainty in log-space (used in EKF)
        - heading_rate: rate of heading change (rad/s)
    """
    def __init__(self, in_ch=6, channels=64, n_layers=5):
        super().__init__()

        # Stack of TCN residual blocks
        layers = []
        ch = in_ch
        for i in range(n_layers):
            layers.append(TCNBlock(ch, channels, dilation=2 ** i))
            ch = channels
        self.tcn = nn.Sequential(*layers)

        # Global average pooling over time dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Output head: channels -> 32 -> 3
        self.head = nn.Sequential(
            nn.Linear(channels, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [mean, log_var, heading_rate]
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch_size, 6, window_size)

        Returns:
            (batch_size, 3) = [speed_mean (km/h), speed_log_var, heading_rate]
        """
        # x: (batch, 6, window)
        h = self.tcn(x)  # (batch, channels, window)
        h = self.pool(h).squeeze(-1)  # (batch, channels)
        return self.head(h)  # (batch, 3)


def build_model():
    """Factory function for building the model."""
    return VelocityTCN(in_ch=6, channels=64, n_layers=5)


def count_parameters(model):
    """Count trainable parameters and estimate model size."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size (FP32): {total * 4 / 1024:.1f} KB")
    print(f"Model size (INT8 estimate): {total / 1024:.1f} KB")
    return total


# Quick self-test
if __name__ == "__main__":
    print("=" * 60)
    print("TCN MODEL SELF-TEST")
    print("=" * 60)

    # Test model
    model = build_model()
    print(f"\n[OK] Model instantiated successfully")

    # Test forward pass
    batch_size = 4
    window_size = 100
    x = torch.randn(batch_size, 6, window_size)

    with torch.no_grad():
        output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output components: [speed_mean, speed_log_var, heading_rate]")
    print(f"Sample output: {output[0].tolist()}")

    # Count parameters
    print("\nParameter count:")
    count_parameters(model)

    print("\n[OK] All tests passed!")

