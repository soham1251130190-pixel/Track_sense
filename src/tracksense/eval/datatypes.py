"""
TrackSense Data Types & Contracts for Evaluation Harness
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class BlackoutInterval:
    """Represents a period during which GNSS signals are unavailable."""
    start_time: float
    end_time: float
    name: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class TrajectoryData:
    """
    Complete ground truth and raw sensor trajectory data (Person 1 contract).
    Positions and velocities are expressed in local ENU frame (meters, m/s).
    """
    timestamps: np.ndarray  # Shape: (N,)
    gt_position: np.ndarray  # Shape: (N, 2) or (N, 3) in ENU meters
    gt_velocity: Optional[np.ndarray] = None  # Shape: (N, 2) or (N, 3) in m/s
    gnss_position: Optional[np.ndarray] = None  # Shape: (N, 2) with NaN during blackouts
    imu_accel: Optional[np.ndarray] = None  # Shape: (N, 3) in m/s^2
    imu_gyro: Optional[np.ndarray] = None  # Shape: (N, 3) in rad/s
    blackout_intervals: List[BlackoutInterval] = field(default_factory=list)

    def __post_init__(self):
        if len(self.timestamps) != len(self.gt_position):
            raise ValueError(
                f"Mismatch: timestamps ({len(self.timestamps)}) vs gt_position ({len(self.gt_position)})"
            )


@dataclass
class PredictionData:
    """
    Predicted trajectory from a model or pipeline (Person 2 / Person 3 contract).
    """
    name: str
    timestamps: np.ndarray  # Shape: (N,)
    predicted_position: np.ndarray  # Shape: (N, 2)
    predicted_velocity: Optional[np.ndarray] = None  # Shape: (N, 2)
    predicted_log_variance: Optional[np.ndarray] = None  # Shape: (N,) uncertainty


@dataclass
class SegmentMetrics:
    """Evaluation metrics computed over a specific GNSS blackout interval."""
    interval: BlackoutInterval
    blackout_duration_s: float
    distance_traveled_m: float
    end_drift_distance_m: float
    drift_percentage: float
    max_position_error_m: float
    rmse_m: float
    mae_m: float
    passed_target: bool


@dataclass
class MetricResults:
    """Overall metric results across an entire trajectory and all blackout segments."""
    prediction_name: str
    overall_rmse_m: float
    overall_mae_m: float
    overall_max_error_m: float
    overall_95th_percentile_m: float
    mean_drift_percentage: float
    max_drift_percentage: float
    total_blackout_distance_m: float
    segment_metrics: List[SegmentMetrics] = field(default_factory=list)


@dataclass
class PassFailStatus:
    """Pass/Fail verification result against the project's target drift requirement (<10%)."""
    passed: bool
    target_drift_pct: float
    actual_max_drift_pct: float
    actual_mean_drift_pct: float
    summary: str
