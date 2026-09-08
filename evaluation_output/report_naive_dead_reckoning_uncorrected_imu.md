# TrackSense Evaluation Report Card

### Result: ❌ **FAIL**

**Pipeline Evaluated**: Naive Dead Reckoning (Uncorrected IMU)  
**Target Drift Ceiling**: <10.0%  
**Max Drift Observed**: 39.07%  
**Mean Drift Observed**: 27.75%  

## Summary
FAIL: One or more blackout segments exceeded target drift ceiling of <10.0%. Max drift observed: 39.07% (Mean: 27.75%).

## Overall Trajectory Performance
| Metric | Value |
|---|---|
| RMSE | 37.64 m |
| MAE | 14.97 m |
| 95th Percentile Error | 91.98 m |
| Maximum Error | 190.71 m |
| Total Blackout Distance | 985.7 m |

## Blackout Segment Breakdown
| Segment | Duration | Distance | End Drift | Drift % | Max Error | Status |
|---|---|---|---|---|---|---|
| Tunnel 1 (500m) | 30.0s | 497.6m | 81.79m | 16.44% | 81.79m | False (FAIL) |
| Tunnel 2 (500m) | 30.0s | 488.2m | 190.71m | 39.07% | 190.71m | False (FAIL) |
