"""
TrackSense Report Generator Engine

Exporters for JSON metrics, CSV per-timestep breakdowns, and Markdown summary report cards.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from tracksense.eval.datatypes import (
    TrajectoryData,
    PredictionData,
    MetricResults,
    PassFailStatus,
)
from tracksense.eval.metrics import EvaluationMetrics


class ReportGenerator:
    """Generates formatted CSV, JSON, and Markdown evaluation reports."""

    @staticmethod
    def export_json(
        metric_results: MetricResults,
        pass_fail: PassFailStatus,
        output_path: str,
    ) -> None:
        """Saves metric results and pass/fail evaluation to a structured JSON file."""
        data = {
            "prediction_name": metric_results.prediction_name,
            "pass_fail": {
                "passed": bool(pass_fail.passed),
                "target_drift_pct": float(pass_fail.target_drift_pct),
                "actual_max_drift_pct": float(pass_fail.actual_max_drift_pct),
                "actual_mean_drift_pct": float(pass_fail.actual_mean_drift_pct),
                "summary": str(pass_fail.summary),
            },
            "overall_metrics": {
                "rmse_m": float(metric_results.overall_rmse_m),
                "mae_m": float(metric_results.overall_mae_m),
                "max_error_m": float(metric_results.overall_max_error_m),
                "percentile_95_m": float(metric_results.overall_95th_percentile_m),
                "mean_drift_pct": float(metric_results.mean_drift_percentage),
                "max_drift_pct": float(metric_results.max_drift_percentage),
                "total_blackout_distance_m": float(metric_results.total_blackout_distance_m),
            },
            "segment_metrics": [
                {
                    "segment_name": str(seg.interval.name),
                    "start_time_s": float(seg.interval.start_time),
                    "end_time_s": float(seg.interval.end_time),
                    "duration_s": float(seg.blackout_duration_s),
                    "distance_traveled_m": float(seg.distance_traveled_m),
                    "end_drift_distance_m": float(seg.end_drift_distance_m),
                    "drift_percentage": float(seg.drift_percentage),
                    "rmse_m": float(seg.rmse_m),
                    "mae_m": float(seg.mae_m),
                    "max_error_m": float(seg.max_position_error_m),
                    "passed_target": bool(seg.passed_target),
                }
                for seg in metric_results.segment_metrics
            ],
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_csv(
        gt_data: TrajectoryData,
        predictions: List[PredictionData],
        output_path: str,
    ) -> None:
        """Saves a per-timestep CSV containing ground truth, predictions, and errors."""
        df_dict: Dict[str, Any] = {
            "timestamp_s": gt_data.timestamps,
            "gt_x_m": gt_data.gt_position[:, 0],
            "gt_y_m": gt_data.gt_position[:, 1],
        }

        is_blackout = np.zeros(len(gt_data.timestamps), dtype=bool)
        for interval in gt_data.blackout_intervals:
            mask = (gt_data.timestamps >= interval.start_time) & (
                gt_data.timestamps <= interval.end_time
            )
            is_blackout |= mask
        df_dict["is_gnss_blackout"] = is_blackout

        for pred in predictions:
            prefix = pred.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            df_dict[f"{prefix}_x_m"] = pred.predicted_position[:, 0]
            df_dict[f"{prefix}_y_m"] = pred.predicted_position[:, 1]
            errors = EvaluationMetrics.compute_position_errors(
                gt_data.gt_position, pred.predicted_position
            )
            df_dict[f"{prefix}_error_m"] = errors

        df = pd.DataFrame(df_dict)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def export_markdown_report(
        metric_results: MetricResults,
        pass_fail: PassFailStatus,
        output_path: str,
    ) -> None:
        """Generates a human-readable Markdown evaluation report card."""
        status_badge = "✅ **PASS**" if pass_fail.passed else "❌ **FAIL**"
        
        md_lines = [
            "# TrackSense Evaluation Report Card",
            "",
            f"### Result: {status_badge}",
            "",
            f"**Pipeline Evaluated**: {metric_results.prediction_name}  ",
            f"**Target Drift Ceiling**: <{pass_fail.target_drift_pct:.1f}%  ",
            f"**Max Drift Observed**: {pass_fail.actual_max_drift_pct:.2f}%  ",
            f"**Mean Drift Observed**: {pass_fail.actual_mean_drift_pct:.2f}%  ",
            "",
            "## Summary",
            f"{pass_fail.summary}",
            "",
            "## Overall Trajectory Performance",
            "| Metric | Value |",
            "|---|---|",
            f"| RMSE | {metric_results.overall_rmse_m:.2f} m |",
            f"| MAE | {metric_results.overall_mae_m:.2f} m |",
            f"| 95th Percentile Error | {metric_results.overall_95th_percentile_m:.2f} m |",
            f"| Maximum Error | {metric_results.overall_max_error_m:.2f} m |",
            f"| Total Blackout Distance | {metric_results.total_blackout_distance_m:.1f} m |",
            "",
            "## Blackout Segment Breakdown",
            "| Segment | Duration | Distance | End Drift | Drift % | Max Error | Status |",
            "|---|---|---|---|---|---|---|",
        ]

        for seg in metric_results.segment_metrics:
            seg_status = "PASS" if seg.passed_target else "FAIL"
            name = seg.interval.name or f"Segment ({seg.interval.start_time:.0f}s - {seg.interval.end_time:.0f}s)"
            md_lines.append(
                f"| {name} | {seg.blackout_duration_s:.1f}s | {seg.distance_traveled_m:.1f}m | "
                f"{seg.end_drift_distance_m:.2f}m | {seg.drift_percentage:.2f}% | {seg.max_position_error_m:.2f}m | {seg.passed_target} ({seg_status}) |"
            )

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
