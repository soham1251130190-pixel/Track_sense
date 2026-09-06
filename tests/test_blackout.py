"""
Unit Tests for Blackout Window Processing
"""

import pytest
import numpy as np
from tracksense.eval.datatypes import BlackoutInterval
from tracksense.eval.metrics import EvaluationMetrics


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
