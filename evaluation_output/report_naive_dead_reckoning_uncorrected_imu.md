# TrackSense Evaluation Report Card

### Result: ❌ **FAIL**

**Pipeline Evaluated**: Naive Dead Reckoning (Uncorrected IMU)  
**Target Drift Ceiling**: <10.0%  
**Max Drift Observed**: 38.92%  
**Mean Drift Observed**: 27.68%  

## Summary
FAIL: One or more blackout segments exceeded target drift ceiling of <10.0%. Max drift observed: 38.92% (Mean: 27.68%).

## Overall Trajectory Performance
| Metric | Value |
|---|---|
| RMSE | 37.34 m |
| MAE | 14.84 m |
| 95th Percentile Error | 90.77 m |
| Maximum Error | 189.97 m |
| Total Blackout Distance | 985.7 m |

## Blackout Segment Breakdown
| Segment | Duration | Distance | End Drift | Drift % | Max Error | Status |
|---|---|---|---|---|---|---|
| Tunnel 1 (500m) | 30.0s | 497.6m | 81.87m | 16.45% | 81.87m | False (FAIL) |
| Tunnel 2 (500m) | 30.0s | 488.2m | 189.97m | 38.92% | 189.97m | False (FAIL) |
