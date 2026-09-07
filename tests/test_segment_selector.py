"""
Unit Tests for Automatic Evaluation Segment Selector (Step 13)
"""

import pytest
import numpy as np
import pandas as pd

from tracksense.eval.datatypes import TrajectoryData, EvaluationSegment
from tracksense.eval.segment_selector import EvaluationSegmentSelector, select_evaluation_segments


@pytest.fixture
def sample_long_trajectory():
    """Generates a 300-second trajectory sampled at 10Hz (3000 rows)."""
    t = np.linspace(0.0, 300.0, 3000)
    # Variable speed curve to test meaningful motion selection
    speed = np.where((t >= 50) & (t <= 200), 20.0, 5.0)
    x = np.cumsum(speed * 0.1)
    y = np.sin(t * 0.05) * 10.0
    gt_pos = np.column_stack([x, y])
    return TrajectoryData(
        timestamps=t,
        gt_position=gt_pos,
    )


@pytest.fixture
def sample_dataframe():
    """Generates a 300-second DataFrame sampled at 10Hz (3000 rows)."""
    t = np.linspace(0.0, 300.0, 3000)
    speed = np.where((t >= 50) & (t <= 200), 20.0, 5.0)
    x = np.cumsum(speed * 0.1)
    y = np.sin(t * 0.05) * 10.0
    return pd.DataFrame({
        "timestamp_s": t,
        "gt_x_m": x,
        "gt_y_m": y,
        "meta": [f"row_{i}" for i in range(len(t))],
    })


def test_select_30s_segment(sample_long_trajectory):
    selector = EvaluationSegmentSelector()
    segments = selector.select_segments(sample_long_trajectory, durations=[30.0])

    assert len(segments) == 1
    seg = segments[0]
    assert isinstance(seg, EvaluationSegment)
    assert seg.duration <= 30.0
    assert seg.duration >= 29.5
    assert seg.start_time >= 0.0
    assert seg.end_time <= 300.0
    assert seg.distance_travelled > 0.0
    assert seg.distance_traveled == seg.distance_travelled  # Alias property test


def test_select_60s_segment(sample_long_trajectory):
    selector = EvaluationSegmentSelector()
    segments = selector.select_segments(sample_long_trajectory, durations=[60.0])

    assert len(segments) == 1
    seg = segments[0]
    assert seg.duration <= 60.0
    assert seg.duration >= 59.5
    assert seg.distance_travelled > 0.0


def test_select_120s_segment(sample_long_trajectory):
    selector = EvaluationSegmentSelector()
    segments = selector.select_segments(sample_long_trajectory, durations=[120.0])

    assert len(segments) == 1
    seg = segments[0]
    assert seg.duration <= 120.0
    assert seg.duration >= 119.5
    assert seg.distance_travelled > 0.0


def test_multiple_durations_non_overlapping(sample_long_trajectory):
    selector = EvaluationSegmentSelector()
    durations = [30.0, 60.0, 120.0]
    segments = selector.select_segments(sample_long_trajectory, durations=durations)

    assert len(segments) == 3
    for s, req_dur in zip(segments, durations):
        assert s.duration <= req_dur
        assert s.duration >= req_dur - 0.5
        assert s.distance_travelled > 0.0

    # Verify non-overlapping
    time_ranges = [(s.start_time, s.end_time) for s in segments]
    for i in range(len(time_ranges)):
        for j in range(i + 1, len(time_ranges)):
            s1_start, s1_end = time_ranges[i]
            s2_start, s2_end = time_ranges[j]
            assert (s1_end < s2_start) or (s2_end < s1_start), f"Segments {i} and {j} overlap!"


def test_deterministic_selection(sample_long_trajectory):
    selector = EvaluationSegmentSelector()
    res1 = selector.select_segments(sample_long_trajectory, durations=[30.0, 60.0, 120.0])
    res2 = selector.select_segments(sample_long_trajectory, durations=[30.0, 60.0, 120.0])

    assert len(res1) == len(res2)
    for s1, s2 in zip(res1, res2):
        assert s1.segment_id == s2.segment_id
        assert s1.start_time == s2.start_time
        assert s1.end_time == s2.end_time
        assert s1.distance_travelled == s2.distance_travelled


def test_dataframe_support(sample_dataframe):
    segments = select_evaluation_segments(sample_dataframe, durations=[30.0, 60.0])
    assert len(segments) == 2
    assert segments[0].distance_travelled > 0.0
    assert segments[1].distance_travelled > 0.0


def test_insufficient_trajectory_duration_error(sample_long_trajectory):
    selector = EvaluationSegmentSelector()
    # Trajectory total duration is 300s. Trying to request 250s + 100s = 350s should fail.
    with pytest.raises(ValueError, match="Insufficient trajectory data"):
        selector.select_segments(sample_long_trajectory, durations=[250.0, 100.0])

    # Single duration exceeding trajectory length
    with pytest.raises(ValueError, match="Insufficient trajectory data"):
        selector.select_segments(sample_long_trajectory, durations=[400.0])


def test_original_trajectory_unchanged(sample_long_trajectory):
    timestamps_orig = np.copy(sample_long_trajectory.timestamps)
    gt_pos_orig = np.copy(sample_long_trajectory.gt_position)

    selector = EvaluationSegmentSelector()
    selector.select_segments(sample_long_trajectory, durations=[30.0, 60.0])

    np.testing.assert_array_equal(sample_long_trajectory.timestamps, timestamps_orig)
    np.testing.assert_array_equal(sample_long_trajectory.gt_position, gt_pos_orig)
    assert len(sample_long_trajectory.blackout_intervals) == 0


def test_invalid_duration_error(sample_long_trajectory):
    selector = EvaluationSegmentSelector()
    with pytest.raises(ValueError, match="must be > 0"):
        selector.select_segments(sample_long_trajectory, durations=[-10.0])

    with pytest.raises(ValueError, match="must be > 0"):
        selector.select_segments(sample_long_trajectory, durations=[0.0])


def test_unordered_timestamps_error():
    t = np.array([0.0, 1.0, 0.5, 3.0])
    gt_pos = np.zeros((4, 2))
    traj = TrajectoryData(timestamps=t, gt_position=gt_pos)

    selector = EvaluationSegmentSelector()
    with pytest.raises(ValueError, match="strictly ordered"):
        selector.select_segments(traj, durations=[1.0])
