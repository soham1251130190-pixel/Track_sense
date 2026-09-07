"""
Unit Tests for Blackout Window Processing & GNSS Blackout Simulator (Step 12)
"""

import pytest
import numpy as np
import pandas as pd
from tracksense.eval.datatypes import BlackoutInterval, TrajectoryData
from tracksense.eval.metrics import EvaluationMetrics
from tracksense.eval.blackout import simulate_gnss_blackout, GNSSBlackoutSimulator


def test_empty_blackout_segment():
    timestamps = np.array([0.0, 1.0, 2.0])
    gt = np.zeros((3, 2))
    pred = np.zeros((3, 2))
    interval = BlackoutInterval(start_time=10.0, end_time=20.0, name="Nonexistent Window")

    metrics = EvaluationMetrics.evaluate_blackout_segment(
        timestamps, gt, pred, interval, target_drift_pct=10.0
    )

    assert metrics.distance_traveled_m == 0.0
    assert metrics.end_drift_distance_m == 0.0
    assert metrics.drift_percentage == 0.0
    assert metrics.passed_target is True


@pytest.fixture
def sample_dataframe():
    """Generates a 200-second sample DataFrame sampled at 10Hz (2000 rows)."""
    t = np.linspace(0.0, 200.0, 2000)
    df = pd.DataFrame({
        "timestamp_s": t,
        "gt_x_m": t * 10.0,
        "gt_y_m": t * 5.0,
        "imu_accel_x": np.sin(t),
        "imu_gyro_z": np.cos(t),
        "pred_x_m": t * 10.0 + 1.0,
        "unrelated_meta_col": [f"id_{i}" for i in range(len(t))],
    })
    return df


@pytest.fixture
def sample_trajectory_data():
    """Generates a 200-second TrajectoryData object sampled at 10Hz."""
    t = np.linspace(0.0, 200.0, 2000)
    gt_pos = np.column_stack([t * 10.0, t * 5.0])
    gt_vel = np.column_stack([np.full_like(t, 10.0), np.full_like(t, 5.0)])
    imu_acc = np.column_stack([np.sin(t), np.zeros_like(t), np.full_like(t, 9.81)])
    return TrajectoryData(
        timestamps=t,
        gt_position=gt_pos,
        gt_velocity=gt_vel,
        imu_accel=imu_acc,
    )


def test_blackout_30s_simulation_dataframe(sample_dataframe):
    df_copy = sample_dataframe.copy()
    out = simulate_gnss_blackout(sample_dataframe, blackout_start_time=20.0, blackout_duration=30.0)

    # 1. 30s blackout works
    blackout_mask = (out["timestamp_s"] >= 20.0) & (out["timestamp_s"] <= 50.0)
    assert np.all(out.loc[blackout_mask, "gnss_available"] == False)
    assert np.all(out.loc[~blackout_mask, "gnss_available"] == True)

    # 2. Ground truth remains intact
    pd.testing.assert_series_equal(out["gt_x_m"], sample_dataframe["gt_x_m"])
    pd.testing.assert_series_equal(out["gt_y_m"], sample_dataframe["gt_y_m"])

    # 3. IMU data remains intact
    pd.testing.assert_series_equal(out["imu_accel_x"], sample_dataframe["imu_accel_x"])

    # 4. Predictions remain intact
    pd.testing.assert_series_equal(out["pred_x_m"], sample_dataframe["pred_x_m"])

    # 5. Unrelated columns preserved
    pd.testing.assert_series_equal(out["unrelated_meta_col"], sample_dataframe["unrelated_meta_col"])

    # 6. Original input is NOT mutated
    pd.testing.assert_frame_equal(sample_dataframe, df_copy)
    assert "gnss_available" not in sample_dataframe.columns


def test_blackout_60s_simulation_dataframe(sample_dataframe):
    out = simulate_gnss_blackout(sample_dataframe, blackout_start_time=50.0, blackout_duration=60.0)

    blackout_mask = (out["timestamp_s"] >= 50.0) & (out["timestamp_s"] <= 110.0)
    assert np.all(out.loc[blackout_mask, "gnss_available"] == False)
    assert np.all(out.loc[~blackout_mask, "gnss_available"] == True)


def test_blackout_120s_simulation_trajectory_data(sample_trajectory_data):
    out = simulate_gnss_blackout(sample_trajectory_data, blackout_start_time=40.0, blackout_duration=120.0)

    # Check blackout interval added
    assert len(out.blackout_intervals) == 1
    bo = out.blackout_intervals[0]
    assert bo.start_time == 40.0
    assert bo.end_time == 160.0
    assert bo.duration == 120.0

    # Ground truth remains intact
    np.testing.assert_array_equal(out.gt_position, sample_trajectory_data.gt_position)
    np.testing.assert_array_equal(out.imu_accel, sample_trajectory_data.imu_accel)

    # GNSS position masked as NaN inside blackout
    mask = (out.timestamps >= 40.0) & (out.timestamps <= 160.0)
    assert np.all(np.isnan(out.gnss_position[mask]))
    assert not np.any(np.isnan(out.gnss_position[~mask]))

    # Input non-mutation check
    assert len(sample_trajectory_data.blackout_intervals) == 0
    assert sample_trajectory_data.gnss_position is None


def test_blackout_validation_errors(sample_dataframe):
    # Invalid duration <= 0
    with pytest.raises(ValueError, match="duration must be > 0"):
        simulate_gnss_blackout(sample_dataframe, blackout_start_time=10.0, blackout_duration=0.0)

    with pytest.raises(ValueError, match="duration must be > 0"):
        simulate_gnss_blackout(sample_dataframe, blackout_start_time=10.0, blackout_duration=-5.0)

    # Start time out of bounds
    with pytest.raises(ValueError, match="outside dataset timestamp range"):
        simulate_gnss_blackout(sample_dataframe, blackout_start_time=-10.0, blackout_duration=30.0)

    with pytest.raises(ValueError, match="outside dataset timestamp range"):
        simulate_gnss_blackout(sample_dataframe, blackout_start_time=300.0, blackout_duration=30.0)

    # Blackout exceeds dataset end
    with pytest.raises(ValueError, match="exceeds maximum available timestamp"):
        simulate_gnss_blackout(sample_dataframe, blackout_start_time=180.0, blackout_duration=50.0)

    # Unordered timestamps
    unordered_df = sample_dataframe.copy()
    unordered_df.loc[5, "timestamp_s"] = 0.1  # Break monotonicity
    with pytest.raises(ValueError, match="strictly ordered"):
        simulate_gnss_blackout(unordered_df, blackout_start_time=10.0, blackout_duration=20.0)

    # Missing timestamp column
    missing_col_df = sample_dataframe.drop(columns=["timestamp_s"])
    with pytest.raises(KeyError, match="Required timestamp column"):
        simulate_gnss_blackout(missing_col_df, blackout_start_time=10.0, blackout_duration=20.0)
