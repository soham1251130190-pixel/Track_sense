# TrackSense Evaluation Report Card

### Result: ❌ **FAIL**

**Pipeline Evaluated**: TrackSense AI+EKF Pipeline  
**Target Drift Ceiling**: <10.0%  
**Max Drift Observed**: 88.75%  
**Mean Drift Observed**: 57.77%  

## Summary
FAIL: One or more blackout segments exceeded target drift ceiling of <10.0%. Max drift observed: 88.75% (Mean: 57.77%).

## Overall Trajectory Performance
| Metric | Value |
|---|---|
| RMSE | 399.66 m |
| MAE | 249.97 m |
| 95th Percentile Error | 978.86 m |
| Maximum Error | 1039.54 m |
| Total Blackout Distance | 1713.7 m |

## Blackout Segment Breakdown
| Segment | Duration | Distance | End Drift | Drift % | Max Error | Status |
|---|---|---|---|---|---|---|
| 30s_blackout | 30.0s | 0.0m | 0.30m | 0.00% | 0.33m | True (PASS) |
| 60s_blackout | 60.0s | 485.7m | 431.11m | 88.75% | 431.94m | False (FAIL) |
| 120s_blackout | 120.0s | 1228.0m | 1038.50m | 84.57% | 1039.31m | False (FAIL) |
