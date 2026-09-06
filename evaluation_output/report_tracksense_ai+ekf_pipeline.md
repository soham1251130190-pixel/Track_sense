# TrackSense Evaluation Report Card

### Result: ✅ **PASS**

**Pipeline Evaluated**: TrackSense AI+EKF Pipeline  
**Target Drift Ceiling**: <10.0%  
**Max Drift Observed**: 4.56%  
**Mean Drift Observed**: 4.29%  

## Summary
PASS: All blackout segments met target drift ceiling of <10.0%. Max drift observed: 4.56% (Mean: 4.29%).

## Overall Trajectory Performance
| Metric | Value |
|---|---|
| RMSE | 10.50 m |
| MAE | 8.26 m |
| 95th Percentile Error | 19.30 m |
| Maximum Error | 22.26 m |
| Total Blackout Distance | 985.7 m |

## Blackout Segment Breakdown
| Segment | Duration | Distance | End Drift | Drift % | Max Error | Status |
|---|---|---|---|---|---|---|
| Tunnel 1 (500m) | 30.0s | 497.6m | 20.00m | 4.02% | 20.00m | True (PASS) |
| Tunnel 2 (500m) | 30.0s | 488.2m | 22.26m | 4.56% | 22.26m | True (PASS) |
