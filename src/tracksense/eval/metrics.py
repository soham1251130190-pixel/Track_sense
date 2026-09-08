"""
TrackSense Core Evaluation Metrics Engine

Provides vector operations and statistical metrics for evaluating dead-reckoning
and sensor-fusion trajectories against ground truth trajectories.
"""

from typing import List, Tuple, Optional
import numpy as np
from tracksense.eval.datatypes import (
    BlackoutInterval,
    TrajectoryData,
    PredictionData,
    SegmentMetrics,
    MetricResults,
    PassFailStatus,
)


class EvaluationMetrics:
    """Mathematical metric utilities for trajectory and blackout evaluation."""

    @staticmethod
    def compute_position_errors(
        gt_position: np.ndarray, pred_position: np.ndarray
    ) -> np.ndarray:
        """
        Calculates timestep-by-timestep 2D Euclidean position errors in meters.
        
        Args:
            gt_position: Shape (N, 2)
            pred_position: Shape (N, 2)
            
        Returns:
            errors: Shape (N,) array of Euclidean distances
        """
        diff = pred_position[:, :2] - gt_position[:, :2]
        return np.linalg.norm(diff, axis=1)

    @staticmethod
    def compute_cumulative_distance(gt_position: np.ndarray) -> np.ndarray:
        """
        Calculates cumulative ground-truth distance traveled along trajectory.
        
        Args:
            gt_position: Shape (N, 2)
            
        Returns:
            cum_dist: Shape (N,) array of cumulative distance in meters
        """
        if len(gt_position) < 2:
            return np.zeros(len(gt_position))
        diffs = np.diff(gt_position[:, :2], axis=0)
        step_dists = np.linalg.norm(diffs, axis=1)
        cum_dist = np.zeros(len(gt_position))
        cum_dist[1:] = np.cumsum(step_dists)
        return cum_dist

    @classmethod
    def evaluate_blackout_segment(
        self,
        timestamps: np.ndarray,
        gt_position: np.ndarray,
        pred_position: np.ndarray,
        interval: BlackoutInterval,
        target_drift_pct: float = 10.0,
    ) -> SegmentMetrics:
        """
        Evaluates positioning performance during a single GNSS blackout window.
        
        Drift percentage is calculated as:
            (Position Error at End of Blackout / Distance Traveled During Blackout) * 100
        """
        mask = (timestamps >= interval.start_time) & (timestamps <= interval.end_time)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            # Fallback if no samples in range
            return SegmentMetrics(
                interval=interval,
                blackout_duration_s=interval.duration,
                distance_traveled_m=0.0,
                end_drift_distance_m=0.0,
                drift_percentage=0.0,
                max_position_error_m=0.0,
                rmse_m=0.0,
                mae_m=0.0,
                passed_target=True,
            )

        segment_gt = gt_position[indices]
        segment_pred = pred_position[indices]

        errors = self.compute_position_errors(segment_gt, segment_pred)
        end_drift_dist = float(errors[-1])

        # Calculate distance traveled during blackout
        cum_dist = self.compute_cumulative_distance(segment_gt)
        distance_traveled = float(cum_dist[-1])

        if distance_traveled > 1e-3:
            drift_pct = float((end_drift_dist / distance_traveled) * 100.0)
        else:
            drift_pct = 0.0

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(errors))
        max_err = float(np.max(errors))
        passed = bool(drift_pct <= float(target_drift_pct))

        return SegmentMetrics(
            interval=interval,
            blackout_duration_s=float(interval.duration),
            distance_traveled_m=distance_traveled,
            end_drift_distance_m=end_drift_dist,
            drift_percentage=drift_pct,
            max_position_error_m=max_err,
            rmse_m=rmse,
            mae_m=mae,
            passed_target=passed,
        )

    @classmethod
    def evaluate_trajectory(
        self,
        gt_data: TrajectoryData,
        pred_data: PredictionData,
        target_drift_pct: float = 10.0,
    ) -> Tuple[MetricResults, PassFailStatus]:
        """
        Evaluates full trajectory accuracy and all blackout segments.
        """
        errors = self.compute_position_errors(
            gt_data.gt_position, pred_data.predicted_position
        )

        overall_rmse = float(np.sqrt(np.mean(errors ** 2)))
        overall_mae = float(np.mean(errors))
        overall_max = float(np.max(errors))
        overall_95th = float(np.percentile(errors, 95))

        segment_results: List[SegmentMetrics] = []
        for interval in gt_data.blackout_intervals:
            seg_metric = self.evaluate_blackout_segment(
                gt_data.timestamps,
                gt_data.gt_position,
                pred_data.predicted_position,
                interval,
                target_drift_pct=target_drift_pct,
            )
            segment_results.append(seg_metric)

        if segment_results:
            drift_pcts = [s.drift_percentage for s in segment_results]
            mean_drift = float(np.mean(drift_pcts))
            max_drift = float(np.max(drift_pcts))
            total_blackout_dist = float(sum(s.distance_traveled_m for s in segment_results))
            all_passed = bool(all(s.passed_target for s in segment_results))
        else:
            mean_drift = 0.0
            max_drift = 0.0
            total_blackout_dist = 0.0
            all_passed = True

        metric_results = MetricResults(
            prediction_name=pred_data.name,
            overall_rmse_m=overall_rmse,
            overall_mae_m=overall_mae,
            overall_max_error_m=overall_max,
            overall_95th_percentile_m=overall_95th,
            mean_drift_percentage=mean_drift,
            max_drift_percentage=max_drift,
            total_blackout_distance_m=total_blackout_dist,
            segment_metrics=segment_results,
        )

        if all_passed:
            summary = (
                f"PASS: All blackout segments met target drift ceiling of <{target_drift_pct:.1f}%. "
                f"Max drift observed: {max_drift:.2f}% (Mean: {mean_drift:.2f}%)."
            )
        else:
            summary = (
                f"FAIL: One or more blackout segments exceeded target drift ceiling of <{target_drift_pct:.1f}%. "
                f"Max drift observed: {max_drift:.2f}% (Mean: {mean_drift:.2f}%)."
            )

        status = PassFailStatus(
            passed=all_passed,
            target_drift_pct=target_drift_pct,
            actual_max_drift_pct=max_drift,
            actual_mean_drift_pct=mean_drift,
            summary=summary,
        )

        return metric_results, status
