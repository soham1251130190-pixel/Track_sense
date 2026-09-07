"""
TrackSense Main Evaluation CLI Runner

Runs the Person 4 evaluation harness against trajectory and prediction datasets,
producing visual charts, JSON/CSV data files, and printing the Pass/Fail evaluation summary.
"""

import sys
import argparse
from pathlib import Path

# Add src to python path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tracksense.eval.harness import EvaluationHarness
from generate_synthetic_data import generate_synthetic_driving_dataset


def main():
    parser = argparse.ArgumentParser(description="TrackSense Evaluation Harness CLI")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_output",
        help="Directory to save evaluation artifacts (plots, CSVs, reports)",
    )
    parser.add_argument(
        "--target-drift",
        type=float,
        default=10.0,
        help="Target drift percentage threshold ceiling (default 10.0%%)",
    )
    args = parser.parse_args()

    print("==================================================================")
    print("         TRACKSENSE INDEPENDENT EVALUATION HARNESS                ")
    print("==================================================================")
    print(f"Target Drift Ceiling Target: <{args.target_drift:.1f}%")

    # Generate synthetic benchmark dataset
    print("\n[1/3] Generating benchmark dataset and baseline predictions...")
    gt_data, predictions = generate_synthetic_driving_dataset()

    # Execute Evaluation Harness
    print(f"\n[2/3] Running Evaluation Harness across {len(gt_data.blackout_intervals)} blackout segments...")
    harness = EvaluationHarness(target_drift_pct=args.target_drift)
    eval_results = harness.run_evaluation(
        gt_data=gt_data,
        predictions=predictions,
        output_dir=args.output_dir,
        include_baselines=True,
    )

    # Print Summary Report
    print(f"\n[3/3] Evaluation artifacts generated in: '{args.output_dir}/'")
    print("\n" + "="*66)
    print("                    EVALUATION RESULTS SUMMARY                    ")
    print("="*66)

    overall_pass = True
    for pred_name, (metrics, status) in eval_results.items():
        status_str = "PASS" if status.passed else "FAIL"
        print(f"\nPipeline: {pred_name}")
        print(f"  Status:               [{status_str}]")
        print(f"  Overall RMSE:         {metrics.overall_rmse_m:.2f} m")
        print(f"  Overall MAE:          {metrics.overall_mae_m:.2f} m")
        print(f"  95th Percentile Error: {metrics.overall_95th_percentile_m:.2f} m")
        print(f"  Max Position Error:   {metrics.overall_max_error_m:.2f} m")
        print(f"  Max Blackout Drift:   {metrics.max_drift_percentage:.2f}% (Target: <{args.target_drift:.1f}%)")
        print(f"  Mean Blackout Drift:  {metrics.mean_drift_percentage:.2f}%")

        if not status.passed:
            overall_pass = False

    print("\n" + "="*66)
    if overall_pass:
        print("OVERALL RESULT: TRACKSENSE PIPELINE PASSED REQUIREMENTS [PASS]")
    else:
        print("OVERALL RESULT: ONE OR MORE PIPELINES FAILED DRIFT CEILING TARGET [FAIL]")
    print("="*66 + "\n")


if __name__ == "__main__":
    main()
