"""
ekf_fusion_with_logging.py
--------------------------
EKF with full innovation logging for proper tuning using NIS.
Extends VehicleEKF to log innovations for tuning analysis.
"""

import numpy as np
from scipy.stats import chi2
from ekf_fusion import VehicleEKF

class VehicleEKFWithLogging(VehicleEKF):
    def __init__(self, initial_state=None, initial_covariance=None):
        super().__init__(initial_state, initial_covariance)
        
        # Store innovations for tuning
        self.gps_innovations = []      # y = z - Hx for GPS
        self.gps_innovation_cov = []   # S = HPH^T + R for GPS
        self.vel_innovations = []      # y for velocity updates
        self.vel_innovation_cov = []   # S for velocity updates
        
        # Store pre-update states for analysis
        self.pre_update_states = []
        self.update_types = []  # 'gps' or 'vel'
        
        self.log_innovations = True
    
    def update_gps(self, px_meas, py_meas):
        """Log GPS innovation before applying update."""
        z = np.array([px_meas, py_meas])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        
        # Compute innovation BEFORE update
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R_gps
        
        if self.log_innovations:
            self.gps_innovations.append(y.copy())
            self.gps_innovation_cov.append(S.copy())
            self.pre_update_states.append(self.x.copy())
            self.update_types.append('gps')
        
        # Now call parent update (which handles gating)
        return super().update_gps(px_meas, py_meas)
    
    def update_velocity(self, v_kmh, heading_meas, velocity_log_variance=None):
        """Log velocity innovation before applying update."""
        v_ms = v_kmh / 3.6
        heading_wrapped = self._wrap_to_pi(heading_meas)
        z = np.array([v_ms, heading_wrapped])
        H = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        
        # Compute R for this update
        if velocity_log_variance is not None:
            velocity_variance_ms2 = np.exp(velocity_log_variance) / (3.6 ** 2)
            R = np.diag([velocity_variance_ms2, self.R_vel[1, 1]])
        else:
            R = self.R_vel
        
        # Compute innovation BEFORE update
        y = z - H @ self.x
        y[1] = self._wrap_to_pi(y[1])  # Wrap heading innovation
        S = H @ self.P @ H.T + R
        
        if self.log_innovations:
            self.vel_innovations.append(y.copy())
            self.vel_innovation_cov.append(S.copy())
            self.pre_update_states.append(self.x.copy())
            self.update_types.append('vel')
        
        return super().update_velocity(v_kmh, heading_meas, velocity_log_variance)
    
    def get_innovation_stats(self):
        """Compute comprehensive innovation statistics for tuning."""
        stats = {}
        
        # GPS Innovation Statistics
        if self.gps_innovations:
            gps_innov = np.array(self.gps_innovations)
            gps_cov = np.array(self.gps_innovation_cov)
            
            # Innovation mean (should be ~0 for unbiased filter)
            stats['gps_mean'] = np.mean(gps_innov, axis=0)
            
            # Actual innovation covariance
            stats['gps_actual_cov'] = np.cov(gps_innov.T)
            
            # Average predicted covariance (from EKF)
            stats['gps_pred_cov'] = np.mean(gps_cov, axis=0)
            
            # Normalized Innovation Squared (NIS)
            nis_values = []
            for y, S in zip(gps_innov, gps_cov):
                try:
                    nis = y.T @ np.linalg.inv(S) @ y
                    nis_values.append(float(nis))
                except:
                    nis_values.append(float('inf'))
            
            stats['gps_nis_mean'] = np.mean(nis_values)
            stats['gps_nis_median'] = np.median(nis_values)
            stats['gps_nis_std'] = np.std(nis_values)
            stats['gps_nis_gt_5_99'] = sum(1 for n in nis_values if n > 5.99) / len(nis_values)
            stats['gps_n_samples'] = len(gps_innov)
            
            # Innovation autocorrelation (should be ~0 for well-tuned filter)
            if len(gps_innov) > 1:
                stats['gps_autocorr_x'] = np.corrcoef(gps_innov[:-1, 0], gps_innov[1:, 0])[0, 1]
                stats['gps_autocorr_y'] = np.corrcoef(gps_innov[:-1, 1], gps_innov[1:, 1])[0, 1]
            else:
                stats['gps_autocorr_x'] = 0
                stats['gps_autocorr_y'] = 0
        
        # Velocity Innovation Statistics
        if self.vel_innovations:
            vel_innov = np.array(self.vel_innovations)
            vel_cov = np.array(self.vel_innovation_cov)
            
            stats['vel_mean'] = np.mean(vel_innov, axis=0)
            stats['vel_actual_cov'] = np.cov(vel_innov.T)
            stats['vel_pred_cov'] = np.mean(vel_cov, axis=0)
            
            # NIS for velocity
            nis_values = []
            for y, S in zip(vel_innov, vel_cov):
                try:
                    nis = y.T @ np.linalg.inv(S) @ y
                    nis_values.append(float(nis))
                except:
                    nis_values.append(float('inf'))
            
            stats['vel_nis_mean'] = np.mean(nis_values)
            stats['vel_nis_median'] = np.median(nis_values)
            stats['vel_n_samples'] = len(vel_innov)
        
        return stats
    
    def get_tuning_recommendations(self):
        """Generate specific tuning recommendations from innovation stats."""
        stats = self.get_innovation_stats()
        recommendations = {}
        
        if 'gps_n_samples' in stats and stats['gps_n_samples'] > 0:
            # R_gps tuning: match actual innovation covariance
            actual_cov = stats['gps_actual_cov']
            current_r = self.R_gps
            
            # Only use diagonal elements (cross-covariance should be 0)
            recommended_r = np.diag([actual_cov[0,0], actual_cov[1,1]])
            
            # Clamp to reasonable values
            recommended_r = np.clip(recommended_r, 0.1, 100.0)
            
            recommendations['r_gps'] = {
                'current': current_r,
                'recommended': recommended_r,
                'ratio': recommended_r[0,0] / current_r[0,0]
            }
            
            # Q tuning based on NIS
            nis_mean = stats['gps_nis_mean']
            if nis_mean > 2.5:
                # Innovations too large → Q too small → increase Q
                q_scale = max(nis_mean / 2.0, 1.0)
                recommendations['q'] = {
                    'current_scale': 1.0,
                    'recommended_scale': q_scale,
                    'reason': f'NIS mean {nis_mean:.3f} > 2.5 → increase process noise'
                }
            elif nis_mean < 1.0:
                # Innovations too small → Q too large → decrease Q
                q_scale = max(nis_mean / 2.0, 0.1)
                recommendations['q'] = {
                    'current_scale': 1.0,
                    'recommended_scale': q_scale,
                    'reason': f'NIS mean {nis_mean:.3f} < 1.0 → decrease process noise'
                }
            else:
                recommendations['q'] = {
                    'current_scale': 1.0,
                    'recommended_scale': 1.0,
                    'reason': f'NIS mean {nis_mean:.3f} in ideal range (1.0-2.5)'
                }
            
            # Check innovation autocorrelation
            if 'gps_autocorr_x' in stats:
                if abs(stats['gps_autocorr_x']) > 0.3 or abs(stats['gps_autocorr_y']) > 0.3:
                    recommendations['warning'] = (
                        f"Innovation autocorrelation detected: x={stats['gps_autocorr_x']:.3f}, "
                        f"y={stats['gps_autocorr_y']:.3f}. This suggests the filter is not "
                        "capturing all dynamics (may need state augmentation)."
                    )
        
        return recommendations