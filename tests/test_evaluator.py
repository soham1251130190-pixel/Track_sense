"""
Integration Test for TrackSense Evaluation Harness
"""

import tempfile
import pytest
from pathlib import Path
from tracksense.eval.harness import EvaluationHarness
from scripts.generate_synthetic_data import generate_synthetic_driving_dataset


def test_full_evaluation_harness():
    gt_data, predictions = generate_synthetic_driving_dataset(duration_s=60.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        harness = EvaluationHarness(target_drift_pct=10.0)
        results = harness.run_evaluation(
            gt_data, predictions, output_dir=tmp_dir, include_baselines=True
        )

        assert "TrackSense AI+EKF Pipeline" in results
        assert "Naive Dead Reckoning (Uncorrected IMU)" in results
        assert "Frozen GNSS Baseline (Standard App)" in results

        # Verify artifacts were saved to output_dir
        out_path = Path(tmp_dir)
        assert (out_path / "trajectory_2d_map.png").exists()
        assert (out_path / "position_error_over_time.png").exists()
        assert (out_path / "drift_percentage_comparison.png").exists()
        assert (out_path / "evaluation_timestep_metrics.csv").exists()

        # TrackSense should pass
        ts_metrics, ts_status = results["TrackSense AI+EKF Pipeline"]
        assert ts_status.passed is True
        assert ts_metrics.max_drift_percentage < 10.0


def test_tcn_prediction_reaches_ekf_as_mps():
    """
    Verifies that a TCN prediction of 10.0 m/s reaches VehicleEKF state as 10.0 m/s
    (not 2.78 m/s or 36 m/s) via adapters.py.
    """
    import torch
    import numpy as np
    from tracksense.eval.adapters import TCN_EKF_ModelAdapter
    from tracksense.eval.datatypes import TrajectoryData

    # 100 timesteps @ 10Hz
    t = np.arange(100) * 0.1
    gt_pos = np.column_stack([np.linspace(0, 100, 100), np.zeros(100)])
    gt_vel = np.column_stack([np.full(100, 10.0), np.zeros(100)])
    imu_accel = np.column_stack([np.zeros(100), np.zeros(100), np.full(100, 9.81)])
    imu_gyro = np.zeros((100, 3))

    gt_data = TrajectoryData(
        timestamps=t,
        gt_position=gt_pos,
        gt_velocity=gt_vel,
        gnss_position=gt_pos,
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        blackout_intervals=[]
    )

    class MockModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            out = torch.zeros((batch_size, 3), dtype=torch.float32)
            out[:, 0] = 10.0  # TCN predicts 10.0 m/s
            return out

    mock_model = MockModel()
    adapter = TCN_EKF_ModelAdapter(model_instance=mock_model)

    pred_data = adapter.predict(gt_data)

    v_norms = np.linalg.norm(pred_data.predicted_velocity, axis=1)
    mean_speed_mps = np.mean(v_norms[10:])

    # Must be close to 10.0 m/s (not 2.78 m/s or 36 m/s)
    assert np.isclose(mean_speed_mps, 10.0, atol=0.5), f"Expected ~10.0 m/s, got {mean_speed_mps:.2f} m/s"


def test_imu_gyro_channel_mapping_yaw_to_matrix_col3():
    """
    Verifies that when imu_gyro is formatted as [gyro_yaw, gyro_pitch, gyro_roll],
    imu_matrix[:, 3] (the yaw rate integrated for heading) receives gyro_yaw (column 0),
    not gyro_roll (column 2).
    """
    import torch
    import numpy as np
    from tracksense.eval.adapters import TCN_EKF_ModelAdapter
    from tracksense.eval.datatypes import TrajectoryData

    N = 10
    t = np.arange(N) * 0.1
    imu_accel = np.zeros((N, 3))
    # Distinct values for yaw (1.23), pitch (4.56), roll (7.89)
    imu_gyro = np.column_stack([np.full(N, 1.23), np.full(N, 4.56), np.full(N, 7.89)])

    traj = TrajectoryData(
        timestamps=t,
        gt_position=np.zeros((N, 2)),
        gt_velocity=np.zeros((N, 2)),
        gnss_position=np.zeros((N, 2)),
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        blackout_intervals=[]
    )

    adapter = TCN_EKF_ModelAdapter(model_instance=torch.nn.Identity())
    imu_matrix = adapter._extract_imu_matrix_from_trajectory(traj)

    # imu_matrix shape: (N, 6) = [ax, ay, az, gz, gy, gx]
    # imu_matrix[:, 3] must contain gyro_yaw (1.23), not gyro_roll (7.89)
    np.testing.assert_allclose(imu_matrix[:, 3], 1.23, err_msg="imu_matrix[:, 3] should contain gyro_yaw (1.23)")
    np.testing.assert_allclose(imu_matrix[:, 5], 7.89, err_msg="imu_matrix[:, 5] should contain gyro_roll (7.89)")


def test_pre_blackout_heading_realigned_from_latest_gnss():
    """
    Regression test: Verifies that if a vehicle changes direction before a blackout,
    the blackout starts with the latest pre-blackout GNSS-derived heading rather than
    the original t=0 heading.
    """
    import torch
    import numpy as np
    from tracksense.eval.adapters import TCN_EKF_ModelAdapter
    from tracksense.eval.datatypes import TrajectoryData, BlackoutInterval

    # 300 timesteps @ 10Hz (30s duration)
    # t=0..10s (idx 0..100): move East along +X: (0,0) -> (100, 0)
    # t=10..20s (idx 100..200): turn and move North along +Y: (100,0) -> (100, 100)
    # t=20..30s (idx 200..300): blackout!
    N = 300
    t = np.arange(N) * 0.1

    gt_pos = np.zeros((N, 2))
    # Segment 1 (East): idx 0..100
    gt_pos[:100, 0] = np.linspace(0, 100, 100)
    gt_pos[:100, 1] = 0.0
    # Segment 2 (North): idx 100..200
    gt_pos[100:200, 0] = 100.0
    gt_pos[100:200, 1] = np.linspace(0, 100, 100)
    # Segment 3 (North blackout): idx 200..300
    gt_pos[200:, 0] = 100.0
    gt_pos[200:, 1] = np.linspace(100, 200, 100)

    gt_vel = np.zeros((N, 2))
    imu_accel = np.column_stack([np.zeros(N), np.zeros(N), np.full(N, 9.81)])
    imu_gyro = np.zeros((N, 3))

    blackouts = [BlackoutInterval(start_time=20.0, end_time=30.0, name="blackout_20_30")]
    gnss_pos = np.copy(gt_pos)
    gnss_pos[200:] = np.nan

    traj = TrajectoryData(
        timestamps=t,
        gt_position=gt_pos,
        gt_velocity=gt_vel,
        gnss_position=gnss_pos,
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        blackout_intervals=blackouts
    )

    class MockModel(torch.nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            out = torch.zeros((batch_size, 3), dtype=torch.float32)
            out[:, 0] = 10.0  # TCN predicts 10.0 m/s
            return out

    adapter = TCN_EKF_ModelAdapter(model_instance=MockModel())
    pred_data = adapter.predict(traj)

    # At blackout start (idx 200, t=20s), direction should be North (+Y, angle ~ +pi/2 = 1.57 rad), NOT East (0.0 rad)
    pred_vel_at_blackout_start = pred_data.predicted_velocity[200]
    heading_at_start = np.arctan2(pred_vel_at_blackout_start[1], pred_vel_at_blackout_start[0])

    # Expected angle is ~pi/2 (+1.57 rad = +90 deg)
    assert np.isclose(heading_at_start, np.pi / 2, atol=0.2), (
        f"Expected heading at blackout start to be North (~{np.pi/2:.2f} rad), but got {heading_at_start:.2f} rad"
    )


