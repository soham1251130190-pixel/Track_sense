# TrackSense Evaluation Report Card

### Result: ❌ **FAIL**

**Pipeline Evaluated**: Naive Dead Reckoning (Uncorrected IMU)  
**Target Drift Ceiling**: <10.0%  
**Max Drift Observed**: 388.22%  
**Mean Drift Observed**: 169.08%  

## Summary
FAIL: One or more blackout segments exceeded target drift ceiling of <10.0%. Max drift observed: 388.22% (Mean: 169.08%).

## Overall Trajectory Performance
| Metric | Value |
|---|---|
| RMSE | 1283.33 m |
| MAE | 623.29 m |
| 95th Percentile Error | 3501.71 m |
| Maximum Error | 4767.33 m |
| Total Blackout Distance | 1713.7 m |

## Blackout Segment Breakdown
| Segment | Duration | Distance | End Drift | Drift % | Max Error | Status |
|---|---|---|---|---|---|---|
| 30s_blackout | 30.0s | 0.0m | 54.72m | 0.00% | 54.72m | True (PASS) |
| 60s_blackout | 60.0s | 485.7m | 578.15m | 119.02% | 578.15m | False (FAIL) |
| 120s_blackout | 120.0s | 1228.0m | 4767.33m | 388.22% | 4767.33m | False (FAIL) |
