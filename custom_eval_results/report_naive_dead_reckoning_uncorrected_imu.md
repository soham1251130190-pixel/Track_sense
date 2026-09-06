# TrackSense Evaluation Report Card

### Result: ❌ **FAIL**

**Pipeline Evaluated**: Naive Dead Reckoning (Uncorrected IMU)  
**Target Drift Ceiling**: <5.0%  
**Max Drift Observed**: 39.25%  
**Mean Drift Observed**: 27.82%  

## Summary
FAIL: One or more blackout segments exceeded target drift ceiling of <5.0%. Max drift observed: 39.25% (Mean: 27.82%).

## Overall Trajectory Performance
| Metric | Value |
|---|---|
| RMSE | 37.89 m |
| MAE | 15.08 m |
| 95th Percentile Error | 93.70 m |
| Maximum Error | 191.58 m |
| Total Blackout Distance | 985.7 m |

## Blackout Segment Breakdown
| Segment | Duration | Distance | End Drift | Drift % | Max Error | Status |
|---|---|---|---|---|---|---|
| Tunnel 1 (500m) | 30.0s | 497.6m | 81.59m | 16.40% | 81.59m | False (FAIL) |
| Tunnel 2 (500m) | 30.0s | 488.2m | 191.58m | 39.25% | 191.58m | False (FAIL) |
