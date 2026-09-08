"""
TrackSense Evaluation Segment Selector (Person 4 - Step 13)

Automatically finds deterministic, valid, non-overlapping evaluation segments
for specified durations (e.g., 30s, 60s, 120s) preferring meaningful vehicle motion.
"""

from typing import List, Union, Optional, Tuple
import numpy as np
import pandas as pd

from tracksense.eval.datatypes import TrajectoryData, EvaluationSegment, BlackoutInterval
from tracksense.eval.metrics import EvaluationMetrics


class EvaluationSegmentSelector:
    """
    Automatic segment selector for evaluation experiments.
    Finds non-overlapping, valid, deterministic evaluation segments.
    """

    def __init__(
        self,
        min_speed_threshold_mps: float = 0.5,
        step_stride_samples: int = 1,
    ):
        self.min_speed_threshold_mps = min_speed_threshold_mps
        self.step_stride_samples = max(1, step_stride_samples)

    def select_segments(
        self,
        data: Union[TrajectoryData, pd.DataFrame],
        durations: List[float] = [30.0, 60.0, 120.0],
        timestamp_col: str = "timestamp_s",
        gt_x_col: str = "gt_x_m",
        gt_y_col: str = "gt_y_m",
    ) -> List[EvaluationSegment]:
        """
        Selects deterministic, non-overlapping evaluation segments for requested durations.
        
        Args:
            data: Input TrajectoryData or pd.DataFrame
            durations: List of requested blackout durations in seconds
            timestamp_col: Column name for timestamps if data is DataFrame
            gt_x_col: Column name for ground truth X position if data is DataFrame
            gt_y_col: Column name for ground truth Y position if data is DataFrame
            
        Returns:
            List of EvaluationSegment dataclasses
            
        Raises:
            ValueError: If timestamps are unordered, duration <= 0, or insufficient data
            KeyError: If required columns missing from DataFrame
        """
        timestamps, gt_pos = self._extract_timestamps_and_positions(
            data, timestamp_col, gt_x_col, gt_y_col
        )
        self._validate_timestamps(timestamps)

        for d in durations:
            if d <= 0:
                raise ValueError(f"Invalid duration {d}s: duration must be > 0.")

        total_traj_duration = float(timestamps[-1] - timestamps[0])

        # Track occupied time intervals to prevent overlapping segments
        occupied_intervals: List[Tuple[float, float]] = []
        selected_segments: List[EvaluationSegment] = []

        # Compute cumulative distance across whole trajectory using existing metric utility
        cum_dist = EvaluationMetrics.compute_cumulative_distance(gt_pos)

        for idx, duration in enumerate(durations):
            segment = self._find_best_segment(
                timestamps=timestamps,
                cum_dist=cum_dist,
                duration=duration,
                occupied_intervals=occupied_intervals,
                segment_id=f"seg_{int(duration)}s_{idx+1}",
                total_traj_duration=total_traj_duration,
            )
            selected_segments.append(segment)
            occupied_intervals.append((segment.start_time, segment.end_time))

        return selected_segments

    def _extract_timestamps_and_positions(
        self,
        data: Union[TrajectoryData, pd.DataFrame],
        timestamp_col: str,
        gt_x_col: str,
        gt_y_col: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(data, TrajectoryData):
            return data.timestamps, data.gt_position
        elif isinstance(data, pd.DataFrame):
            if timestamp_col not in data.columns:
                raise KeyError(f"Required timestamp column '{timestamp_col}' missing from DataFrame.")
            if gt_x_col not in data.columns or gt_y_col not in data.columns:
                raise KeyError(f"Required position columns '{gt_x_col}', '{gt_y_col}' missing from DataFrame.")
            timestamps = data[timestamp_col].to_numpy()
            gt_pos = data[[gt_x_col, gt_y_col]].to_numpy()
            return timestamps, gt_pos
        else:
            raise TypeError(f"Unsupported data type {type(data)}: must be TrajectoryData or pd.DataFrame.")

    def _validate_timestamps(self, timestamps: np.ndarray):
        if len(timestamps) < 2:
            raise ValueError("Trajectory contains insufficient timesteps (fewer than 2).")
        diffs = np.diff(timestamps)
        if np.any(diffs <= 0):
            raise ValueError("Timestamps are not strictly ordered (monotonically increasing).")

    def _find_best_segment(
        self,
        timestamps: np.ndarray,
        cum_dist: np.ndarray,
        duration: float,
        occupied_intervals: List[Tuple[float, float]],
        segment_id: str,
        total_traj_duration: float,
    ) -> EvaluationSegment:
        N = len(timestamps)
        candidates = []

        max_dt = float(np.max(np.diff(timestamps)))
        min_acceptable_duration = max(0.0, duration - max(1.0, 3.0 * max_dt))

        # Find all valid start indices i where a segment of duration <= requested_duration can fit
        for i in range(0, N, self.step_stride_samples):
            t_start = timestamps[i]
            target_t_end = t_start + duration

            if t_start + min_acceptable_duration > timestamps[-1] + 1e-6:
                break

            # Find matching end index j (last sample where timestamp <= target_t_end)
            j = int(np.searchsorted(timestamps, target_t_end + 1e-9, side="right") - 1)
            if j <= i:
                continue

            t_end = timestamps[j]
            actual_duration = float(t_end - t_start)

            # Ensure actual_duration does not exceed requested duration, but is close to requested duration
            if actual_duration > duration + 1e-6 or actual_duration < min_acceptable_duration:
                continue

            # Check overlap with existing occupied intervals
            overlaps = False
            for occ_start, occ_end in occupied_intervals:
                if not (t_end < occ_start or t_start > occ_end):
                    overlaps = True
                    break

            if not overlaps:
                dist_travelled = float(cum_dist[j] - cum_dist[i])
                candidates.append((dist_travelled, i, j, float(t_start), float(t_end), actual_duration))

        if not candidates:
            raise ValueError(
                f"Insufficient trajectory data to select a non-overlapping segment of duration {duration}s. "
                f"Total trajectory duration is {total_traj_duration:.1f}s."
            )

        # Select candidate with maximum distance travelled (meaningful vehicle movement)
        # Ties are broken deterministically by picking the first occurrence
        best_candidate = max(candidates, key=lambda c: c[0])
        dist_travelled, start_idx, end_idx, start_t, end_t, dur = best_candidate

        return EvaluationSegment(
            segment_id=segment_id,
            start_time=float(start_t),
            end_time=float(end_t),
            duration=float(dur),
            start_index=int(start_idx),
            end_index=int(end_idx),
            distance_travelled=float(dist_travelled),
        )


def select_evaluation_segments(
    data: Union[TrajectoryData, pd.DataFrame],
    durations: List[float] = [30.0, 60.0, 120.0],
    timestamp_col: str = "timestamp_s",
    gt_x_col: str = "gt_x_m",
    gt_y_col: str = "gt_y_m",
) -> List[EvaluationSegment]:
    """Convenience wrapper for EvaluationSegmentSelector().select_segments."""
    selector = EvaluationSegmentSelector()
    return selector.select_segments(
        data=data,
        durations=durations,
        timestamp_col=timestamp_col,
        gt_x_col=gt_x_col,
        gt_y_col=gt_y_col,
    )
