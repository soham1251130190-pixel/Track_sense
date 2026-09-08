"""
TrackSense Evaluation Harness Orchestrator

Main entrypoint class for Person 4 to run comprehensive evaluations on ground truth
and model/pipeline prediction trajectories.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from tracksense.eval.datatypes import (
    TrajectoryData,
    PredictionData,
    MetricResults,
    PassFailStatus,
)
from tracksense.eval.metrics import EvaluationMetrics
from tracksense.eval.baselines import BaselineModels
from tracksense.eval.report import ReportGenerator
from tracksense.eval.plotting import PlotGenerator


class EvaluationHarness:
    """
    Main evaluation pipeline engine for TrackSense.
    
    Accepts:
        - Ground truth trajectory + IMU + GNSS + blackout windows (Person 1 contract)
        - Predicted trajectories (Person 2 / Person 3 contracts)
        
    Produces:
        - Position error, RMSE, MAE, 95th percentile
        - Blackout segment drift distance & percentage metrics
        - Baseline comparisons (Naive DR, Frozen GNSS)
        - Plot figures (2D trajectory maps, error over time, drift bar chart)
        - CSV/JSON evaluation reports
        - Pass/Fail verification against <10% drift target
    """

    def __init__(self, target_drift_pct: float = 10.0):
        self.target_drift_pct = target_drift_pct

    def run_evaluation(
        self,
        gt_data: TrajectoryData,
        predictions: List[PredictionData],
        output_dir: Optional[str] = None,
        include_baselines: bool = True,
    ) -> Dict[str, Tuple[MetricResults, PassFailStatus]]:
        """
        Executes full evaluation on provided predictions, optionally adding default baselines.
        
        Args:
            gt_data: TrajectoryData object
            predictions: List of PredictionData objects to evaluate
            output_dir: Path to directory for saving plots, CSVs, JSONs, and Markdown reports
            include_baselines: Whether to automatically compute Naive DR & Frozen GNSS baselines
            
        Returns:
            Dictionary mapping prediction_name -> (MetricResults, PassFailStatus)
        """
        all_preds = list(predictions)

        if include_baselines:
            naive_dr = BaselineModels.run_naive_dead_reckoning(gt_data)
            frozen_gnss = BaselineModels.run_frozen_gnss_baseline(gt_data)
            all_preds.extend([naive_dr, frozen_gnss])

        eval_results: Dict[str, Tuple[MetricResults, PassFailStatus]] = {}
        metric_results_list: List[MetricResults] = []

        for pred in all_preds:
            metric_results, pass_fail = EvaluationMetrics.evaluate_trajectory(
                gt_data, pred, target_drift_pct=self.target_drift_pct
            )
            eval_results[pred.name] = (metric_results, pass_fail)
            metric_results_list.append(metric_results)

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Generate Plots
            PlotGenerator.plot_2d_trajectories(
                gt_data,
                all_preds,
                str(out_path / "trajectory_2d_map.png"),
            )
            PlotGenerator.plot_error_over_time(
                gt_data,
                all_preds,
                str(out_path / "position_error_over_time.png"),
            )
            PlotGenerator.plot_drift_bar_chart(
                metric_results_list,
                str(out_path / "drift_percentage_comparison.png"),
                target_drift_pct=self.target_drift_pct,
            )

            # Export Per-Timestep CSV
            ReportGenerator.export_csv(
                gt_data, all_preds, str(out_path / "evaluation_timestep_metrics.csv")
            )

            # Export JSON & Markdown for each evaluated pipeline
            for pred_name, (m_res, pf_stat) in eval_results.items():
                safe_name = pred_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                ReportGenerator.export_json(
                    m_res, pf_stat, str(out_path / f"metrics_{safe_name}.json")
                )
                ReportGenerator.export_markdown_report(
                    m_res, pf_stat, str(out_path / f"report_{safe_name}.md")
                )

        return eval_results
