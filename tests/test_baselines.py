"""
Unit Tests for Baseline Algorithms (Person 4 - Step 14)
"""

import pytest
import numpy as np
import pandas as pd
from tracksense.eval.datatypes import TrajectoryData, BlackoutInterval, PredictionData
from tracksense.eval.baselines import naive_dead_reckoning, frozen_gnss, BaselineModels


@pytest.fixture
def straight_line_trajectory():
    """100-second straight-line trajectory at 10 m/s (1000 rows @ 10Hz) with 30s blackout."""
    t = np.linspace(0.0, 100.0, 1000)
    speed = 10.0
    gt_x = speed * t
    gt_y = np.zeros_like(t)
    gt_pos = np.column_stack([gt_x, gt_y])
    gt_vel = np.column_stack([np.full_like(t, speed), np.zeros_like(t)])

    # Blackout between 20s and 50s
    blackout = BlackoutInterval(start_time=20.0, end_time=50.0, name="Tunnel_30s")
    gnss_pos = np.copy(gt_pos)
    mask = (t >= 20.0) & (t <= 50.0)
    gnss_pos[mask] = np.nan

    imu_accel = np.zeros((1000, 3))  # Zero acceleration for constant speed

    trj = TrajectoryData(
        timestamps=t,
        gt_position=gt_pos,
        gt_velocity=gt_vel,
        gnss_position=gnss_pos,
        imu_accel=imu_accel,
        blackout_intervals=[blackout],
    )
    return trj


@pytest.fixture
def straight_line_dataframe():
    """DataFrame equivalent of the 100-second straight-line trajectory."""
    t = np.linspace(0.0, 100.0, 1000)
    speed = 10.0
    gt_x = speed * t
    gt_y = np.zeros_like(t)

    gnss_x = np.copy(gt_x)
    gnss_y = np.copy(gt_y)
    mask = (t >= 20.0) & (t <= 50.0)
    gnss_x[mask] = np.nan
    gnss_y[mask] = np.nan

    df = pd.DataFrame({
        "timestamp_s": t,
        "gt_x_m": gt_x,
        "gt_y_m": gt_y,
        "gnss_x_m": gnss_x,
        "gnss_y_m": gnss_y,
        "gnss_available": ~mask,
        "imu_accel_x": np.zeros_like(t),
        "imu_accel_y": np.zeros_like(t),
        "meta_info": [f"idx_{i}" for i in range(len(t))],
    })
    return df


def test_naive_dr_deterministic(straight_line_trajectory):
    res1 = naive_dead_reckoning(straight_line_trajectory)
    res2 = naive_dead_reckoning(straight_line_trajectory)

    assert isinstance(res1, PredictionData)
    assert isinstance(res2, PredictionData)
    np.testing.assert_array_equal(res1.predicted_position, res2.predicted_position)
    np.testing.assert_array_equal(res1.predicted_velocity, res2.predicted_velocity)


def test_frozen_gnss_remains_fixed_during_blackout(straight_line_trajectory):
    res = frozen_gnss(straight_line_trajectory)
    t = straight_line_trajectory.timestamps

    # Find blackout timestamps (20s to 50s)
    mask = (t >= 20.0) & (t <= 50.0)
    bo_positions = res.predicted_position[mask]

    # Position at 20s start of blackout
    start_pos = bo_positions[0]

    # All positions inside blackout must equal start_pos
    for pos in bo_positions:
        np.testing.assert_allclose(pos, start_pos, atol=1e-6)

    # Velocity inside blackout must be zero
    bo_velocities = res.predicted_velocity[mask]
    np.testing.assert_allclose(bo_velocities, 0.0, atol=1e-6)


def test_baselines_do_not_use_ground_truth(straight_line_trajectory):
    # Modify GT position inside blackout completely
    trj_modified = TrajectoryData(
        timestamps=np.copy(straight_line_trajectory.timestamps),
        gt_position=straight_line_trajectory.gt_position + 1000.0,  # Huge shift in GT
        gt_velocity=straight_line_trajectory.gt_velocity,
        gnss_position=np.copy(straight_line_trajectory.gnss_position),
        imu_accel=np.copy(straight_line_trajectory.imu_accel),
        blackout_intervals=straight_line_trajectory.blackout_intervals,
    )

    res_orig_ndr = naive_dead_reckoning(straight_line_trajectory)
    res_mod_ndr = naive_dead_reckoning(trj_modified)

    # Predictions must be identical because baseline uses GNSS/IMU, NOT ground truth
    np.testing.assert_allclose(res_orig_ndr.predicted_position, res_mod_ndr.predicted_position)

    res_orig_fg = frozen_gnss(straight_line_trajectory)
    res_mod_fg = frozen_gnss(trj_modified)
    np.testing.assert_allclose(res_orig_fg.predicted_position, res_mod_fg.predicted_position)


def test_no_future_gnss_used(straight_line_trajectory):
    # Modify future GNSS measurements after blackout (t > 50s)
    gnss_future_mod = np.copy(straight_line_trajectory.gnss_position)
    future_mask = straight_line_trajectory.timestamps > 50.0
    gnss_future_mod[future_mask] += 500.0

    trj_mod = TrajectoryData(
        timestamps=np.copy(straight_line_trajectory.timestamps),
        gt_position=np.copy(straight_line_trajectory.gt_position),
        gt_velocity=straight_line_trajectory.gt_velocity,
        gnss_position=gnss_future_mod,
        imu_accel=straight_line_trajectory.imu_accel,
        blackout_intervals=straight_line_trajectory.blackout_intervals,
    )

    res_orig = naive_dead_reckoning(straight_line_trajectory)
    res_mod = naive_dead_reckoning(trj_mod)

    # Inside blackout (t <= 50s), predictions must be identical regardless of future GNSS
    bo_mask = straight_line_trajectory.timestamps <= 50.0
    np.testing.assert_allclose(
        res_orig.predicted_position[bo_mask], res_mod.predicted_position[bo_mask]
    )


def test_output_timestamps_and_length_match(straight_line_trajectory):
    res_ndr = naive_dead_reckoning(straight_line_trajectory)
    res_fg = frozen_gnss(straight_line_trajectory)

    assert len(res_ndr.timestamps) == len(straight_line_trajectory.timestamps)
    assert len(res_fg.timestamps) == len(straight_line_trajectory.timestamps)
    np.testing.assert_array_equal(res_ndr.timestamps, straight_line_trajectory.timestamps)
    np.testing.assert_array_equal(res_fg.timestamps, straight_line_trajectory.timestamps)


def test_straight_line_expected_behavior(straight_line_trajectory):
    res_ndr = naive_dead_reckoning(straight_line_trajectory)
    t = straight_line_trajectory.timestamps
    bo_mask = (t >= 20.0) & (t <= 50.0)

    # In a constant speed straight-line trajectory with zero accel,
    # Naive DR maintains constant velocity (10 m/s) and continues smoothly
    diffs = np.diff(res_ndr.predicted_position[bo_mask, 0])
    dt = t[1] - t[0]
    expected_step = 10.0 * dt
    np.testing.assert_allclose(diffs, expected_step, atol=1e-3)


def test_dataframe_support(straight_line_dataframe):
    df_ndr = naive_dead_reckoning(straight_line_dataframe)
    df_fg = frozen_gnss(straight_line_dataframe)

    assert isinstance(df_ndr, pd.DataFrame)
    assert isinstance(df_fg, pd.DataFrame)
    assert len(df_ndr) == len(straight_line_dataframe)
    assert "pred_x_m" in df_ndr.columns
    assert "pred_y_m" in df_ndr.columns

    # Verify frozen GNSS stays fixed during blackout in DataFrame
    t = straight_line_dataframe["timestamp_s"].to_numpy()
    bo_mask = (t >= 20.0) & (t <= 50.0)
    bo_x = df_fg.loc[bo_mask, "pred_x_m"].to_numpy()
    np.testing.assert_allclose(bo_x, bo_x[0])


def test_input_data_not_mutated(straight_line_trajectory, straight_line_dataframe):
    gt_copy = np.copy(straight_line_trajectory.gt_position)
    df_copy = straight_line_dataframe.copy()

    naive_dead_reckoning(straight_line_trajectory)
    frozen_gnss(straight_line_dataframe)

    np.testing.assert_array_equal(straight_line_trajectory.gt_position, gt_copy)
    pd.testing.assert_frame_equal(straight_line_dataframe, df_copy)


def test_validation_errors():
    # Unordered timestamps
    t_bad = np.array([0.0, 1.0, 0.5, 3.0])
    gt_pos = np.zeros((4, 2))
    trj_bad = TrajectoryData(timestamps=t_bad, gt_position=gt_pos)

    with pytest.raises(ValueError, match="strictly ordered"):
        naive_dead_reckoning(trj_bad)

    with pytest.raises(ValueError, match="strictly ordered"):
        frozen_gnss(trj_bad)

    # Missing timestamp column in DataFrame
    df_bad = pd.DataFrame({"gnss_x_m": [0, 1], "gnss_y_m": [0, 1]})
    with pytest.raises(KeyError, match="Required timestamp column"):
        naive_dead_reckoning(df_bad)

    # Blackout at t=0 with no valid GNSS fix
    t = np.linspace(0, 10, 100)
    gnss_nan = np.full((100, 2), np.nan)
    trj_nan = TrajectoryData(timestamps=t, gt_position=np.zeros((100, 2)), gnss_position=gnss_nan)
    with pytest.raises(ValueError, match="No valid GNSS fix"):
        naive_dead_reckoning(trj_nan)
