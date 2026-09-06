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
