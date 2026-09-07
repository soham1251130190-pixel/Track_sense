"""
TrackSense Plotting & Visualization Engine

Generates 2D map trajectories, error-over-time plots with shaded blackout zones,
and comparative drift percentage charts.
"""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from tracksense.eval.datatypes import TrajectoryData, PredictionData, MetricResults
from tracksense.eval.metrics import EvaluationMetrics


class PlotGenerator:
    """Generates visual evaluation figures and saves them to file."""

    @staticmethod
    def setup_style():
        """Sets modern clean plotting aesthetic."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["axes.linewidth"] = 0.8

    @classmethod
    def plot_2d_trajectories(
        self,
        gt_data: TrajectoryData,
        predictions: List[PredictionData],
        output_path: str,
        title: str = "TrackSense Vehicle Trajectory & Dead-Reckoning Evaluation",
    ) -> None:
        """
        Plots 2D spatial trajectory map (X vs Y in local ENU meters).
        Highlights GNSS blackout segments with red shaded bounding boxes.
        """
        self.setup_style()
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

        # Plot Ground Truth
        ax.plot(
            gt_data.gt_position[:, 0],
            gt_data.gt_position[:, 1],
            label="Ground Truth Trajectory",
            color="black",
            linewidth=2.5,
            linestyle="-",
            zorder=5,
        )

        # Plot GNSS readings if available
        if gt_data.gnss_position is not None:
            valid_gnss = ~np.isnan(gt_data.gnss_position[:, 0])
            ax.scatter(
                gt_data.gnss_position[valid_gnss, 0],
                gt_data.gnss_position[valid_gnss, 1],
                label="GNSS Measurements",
                color="#0072B2",
                s=12,
                alpha=0.6,
                zorder=3,
            )

        # Color palette for predictions
        colors = ["#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"]
        for idx, pred in enumerate(predictions):
            color = colors[idx % len(colors)]
            ax.plot(
                pred.predicted_position[:, 0],
                pred.predicted_position[:, 1],
                label=pred.name,
                color=color,
                linewidth=1.8,
                linestyle="--",
                zorder=4,
            )

        # Highlight blackout segments
        for interval in gt_data.blackout_intervals:
            mask = (gt_data.timestamps >= interval.start_time) & (
                gt_data.timestamps <= interval.end_time
            )
            if np.any(mask):
                blackout_gt = gt_data.gt_position[mask]
                ax.plot(
                    blackout_gt[:, 0],
                    blackout_gt[:, 1],
                    color="#D55E00",
                    linewidth=4.0,
                    alpha=0.3,
                    zorder=2,
                )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("East / Local X (meters)", fontsize=11)
        ax.set_ylabel("North / Local Y (meters)", fontsize=11)
        ax.axis("equal")
        ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close(fig)

    @classmethod
    def plot_error_over_time(
        self,
        gt_data: TrajectoryData,
        predictions: List[PredictionData],
        output_path: str,
        title: str = "Position Error Over Time & Blackout Intervals",
    ) -> None:
        """
        Plots timestep-by-timestep position error with shaded blackout windows.
        """
        self.setup_style()
        fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

        # Shade blackout intervals
        first_bo = True
        for interval in gt_data.blackout_intervals:
            label = "GNSS Blackout Period" if first_bo else ""
            ax.axvspan(
                interval.start_time,
                interval.end_time,
                color="#FF9999",
                alpha=0.35,
                label=label,
                zorder=1,
            )
            first_bo = False

        colors = ["#D55E00", "#009E73", "#CC79A7", "#0072B2"]
        for idx, pred in enumerate(predictions):
            color = colors[idx % len(colors)]
            errors = EvaluationMetrics.compute_position_errors(
                gt_data.gt_position, pred.predicted_position
            )
            ax.plot(
                gt_data.timestamps,
                errors,
                label=f"{pred.name} (RMSE: {np.sqrt(np.mean(errors**2)):.1f}m)",
                color=color,
                linewidth=1.8,
                zorder=3,
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Timestamp (seconds)", fontsize=11)
        ax.set_ylabel("Position Error (meters)", fontsize=11)
        ax.legend(loc="upper left", frameon=True, facecolor="white")
        plt.tight_layout()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close(fig)

    @classmethod
    def plot_drift_bar_chart(
        self,
        metric_results_list: List[MetricResults],
        output_path: str,
        target_drift_pct: float = 10.0,
    ) -> None:
        """
        Plots comparative bar chart of drift percentage across model pipelines.
        Includes horizontal dashed line for the 10% drift target threshold.
        """
        self.setup_style()
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

        names = [r.prediction_name for r in metric_results_list]
        max_drifts = [r.max_drift_percentage for r in metric_results_list]
        mean_drifts = [r.mean_drift_percentage for r in metric_results_list]

        x = np.arange(len(names))
        width = 0.35

        rects1 = ax.bar(x - width/2, max_drifts, width, label="Max Drift %", color="#D55E00")
        rects2 = ax.bar(x + width/2, mean_drifts, width, label="Mean Drift %", color="#009E73")

        # 10% Target Threshold Line
        ax.axhline(
            y=target_drift_pct,
            color="red",
            linestyle="--",
            linewidth=2.0,
            label=f"Project Target Ceiling ({target_drift_pct:.0f}%)",
            zorder=5,
        )

        ax.set_title("Dead-Reckoning Drift Percentage Comparison", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Drift Percentage (%)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
        ax.legend(loc="upper right", frameon=True)

        # Add bar data labels
        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close(fig)
