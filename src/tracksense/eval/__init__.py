"""
TrackSense Evaluation Module (Person 4)

Independent evaluation harness for dead-reckoning positioning accuracy,
drift percentage calculation, blackout-segment metrics, baseline comparison,
visualization, and report generation.
"""

from tracksense.eval.datatypes import (
    BlackoutInterval,
    TrajectoryData,
    PredictionData,
    MetricResults,
    SegmentMetrics,
    PassFailStatus,
)
from tracksense.eval.metrics import EvaluationMetrics
from tracksense.eval.harness import EvaluationHarness
from tracksense.eval.blackout import GNSSBlackoutSimulator, simulate_gnss_blackout

__all__ = [
    "BlackoutInterval",
    "TrajectoryData",
    "PredictionData",
    "MetricResults",
    "SegmentMetrics",
    "PassFailStatus",
    "EvaluationMetrics",
    "EvaluationHarness",
    "GNSSBlackoutSimulator",
    "simulate_gnss_blackout",
]
