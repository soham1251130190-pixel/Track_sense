# TrackSense Evaluation Report Card

### Result: ❌ **FAIL**

**Pipeline Evaluated**: Frozen GNSS Baseline (Standard App)  
**Target Drift Ceiling**: <10.0%  
**Max Drift Observed**: 88.98%  
**Mean Drift Observed**: 57.86%  

## Summary
FAIL: One or more blackout segments exceeded target drift ceiling of <10.0%. Max drift observed: 88.98% (Mean: 57.86%).

## Overall Trajectory Performance
| Metric | Value |
|---|---|
| RMSE | 397.34 m |
| MAE | 247.12 m |
| 95th Percentile Error | 978.85 m |
| Maximum Error | 1038.99 m |
| Total Blackout Distance | 1713.7 m |

## Blackout Segment Breakdown
| Segment | Duration | Distance | End Drift | Drift % | Max Error | Status |
|---|---|---|---|---|---|---|
| 30s_blackout | 30.0s | 0.0m | 0.00m | 0.00% | 0.00m | True (PASS) |
| 60s_blackout | 60.0s | 485.7m | 432.21m | 88.98% | 432.21m | False (FAIL) |
| 120s_blackout | 120.0s | 1228.0m | 1038.99m | 84.61% | 1038.99m | False (FAIL) |
