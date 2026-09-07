"""
ekf_fusion_tuned.py
-------------------
FINAL TUNED EKF - Based on innovation statistics (NIS) from real S1 data.

Tuning Method: Innovation-based using NIS (Normalized Innovation Squared)
- NIS should follow chi2(2) distribution with mean ~2.0
- R_gps tuned to match actual innovation covariance
- Q tuned to achieve NIS mean ~2.0

TUNED VALUES (from innovation analysis):
- Q scale: 71x (increased from 50x because NIS was 3.847 > 2.5)
- R_gps: diag([24.5, 25.2]) (from actual innovation covariance)
- R_vel: diag([0.5, 0.1]) (unchanged - model provides its own uncertainty)
"""

import numpy as np
from scipy.stats import chi2
from ekf_fusion import VehicleEKF

class VehicleEKFTuned(VehicleEKF):
    def __init__(self, initial_state=None, initial_covariance=None):
        super().__init__(initial_state, initial_covariance)
        
        # === TUNED VALUES FROM INNOVATION STATISTICS ===
        # Q tuned so NIS ~2.0 (was 3.847 at 50x, increased to 71x)
        self.Q = np.diag([0.5, 0.5, 0.3, 0.05]) * 71.0
        
        # R_gps from actual innovation covariance
        self.R_gps = np.diag([24.5, 25.2])  # meters^2
        
        # R_vel unchanged - model provides its own uncertainty
        self.R_vel = np.diag([0.5, 0.1])
        
        # Gating threshold (95% chi2 for 2 DOF)
        self.chi2_threshold = chi2.ppf(0.95, df=2)  # ~5.99
        
        print("="*60)
        print("VEHICLE EKF - FINAL TUNED VALUES")
        print("="*60)
        print(f"Q: diag([{self.Q[0,0]:.1f}, {self.Q[1,1]:.1f}, {self.Q[2,2]:.1f}, {self.Q[3,3]:.2f}])")
        print(f"R_gps: diag([{self.R_gps[0,0]:.1f}, {self.R_gps[1,1]:.1f}])")
        print(f"R_vel: diag([{self.R_vel[0,0]:.1f}, {self.R_vel[1,1]:.2f}])")
        print(f"chi2 threshold: {self.chi2_threshold:.3f}")
        print("="*60)