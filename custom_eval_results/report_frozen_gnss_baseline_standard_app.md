# TrackSense Evaluation Report Card

### Result: ❌ **FAIL**

**Pipeline Evaluated**: Frozen GNSS Baseline (Standard App)  
**Target Drift Ceiling**: <5.0%  
**Max Drift Observed**: 100.34%  
**Mean Drift Observed**: 100.33%  

## Summary
FAIL: One or more blackout segments exceeded target drift ceiling of <5.0%. Max drift observed: 100.34% (Mean: 100.33%).

## Overall Trajectory Performance
| Metric | Value |
|---|---|
| RMSE | 165.33 m |
| MAE | 82.74 m |
| 95th Percentile Error | 420.53 m |
| Maximum Error | 499.27 m |
| Total Blackout Distance | 985.7 m |

## Blackout Segment Breakdown
| Segment | Duration | Distance | End Drift | Drift % | Max Error | Status |
|---|---|---|---|---|---|---|
| Tunnel 1 (500m) | 30.0s | 497.6m | 499.27m | 100.34% | 499.27m | False (FAIL) |
| Tunnel 2 (500m) | 30.0s | 488.2m | 489.76m | 100.33% | 489.76m | False (FAIL) |
