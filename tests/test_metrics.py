"""
Unit Tests for Evaluation Metrics (Person 4)
"""

import pytest
import numpy as np
from tracksense.eval.datatypes import BlackoutInterval
from tracksense.eval.metrics import EvaluationMetrics


def test_position_error():
    gt = np.array([[0.0, 0.0], [3.0, 4.0]])
    pred = np.array([[0.0, 0.0], [0.0, 0.0]])
    errors = EvaluationMetrics.compute_position_errors(gt, pred)
    assert np.isclose(errors[0], 0.0)
    assert np.isclose(errors[1], 5.0)


def test_cumulative_distance():
    gt = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
    cum_dist = EvaluationMetrics.compute_cumulative_distance(gt)
    assert np.isclose(cum_dist[0], 0.0)
    assert np.isclose(cum_dist[1], 5.0)
    assert np.isclose(cum_dist[2], 10.0)


def test_blackout_segment_drift_percentage():
    # 10 timesteps, total 100m distance traveled
    timestamps = np.linspace(0, 10, 10)
    gt = np.column_stack([np.linspace(0, 100, 10), np.zeros(10)])
    # Pred has 5m error at end of blackout
    pred = np.copy(gt)
    pred[-1, 1] = 5.0

    interval = BlackoutInterval(start_time=0.0, end_time=10.0, name="Test Tunnel")
    metrics = EvaluationMetrics.evaluate_blackout_segment(
        timestamps, gt, pred, interval, target_drift_pct=10.0
    )

    assert np.isclose(metrics.distance_traveled_m, 100.0)
    assert np.isclose(metrics.end_drift_distance_m, 5.0)
    assert np.isclose(metrics.drift_percentage, 5.0)
    assert metrics.passed_target is True
