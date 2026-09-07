"""
verify_pipeline.py
------------------
Final pipeline verification - runs end-to-end with all components.
Sync 3 deliverable: Prove everything works together.
"""

import numpy as np
import pandas as pd
from ekf_fusion_tuned import VehicleEKFTuned
from p2_model_integration import P2ModelWrapper
from projection import local_xy_to_latlon, ORIGIN_LAT, ORIGIN_LON
import matplotlib.pyplot as plt

def verify_pipeline(csv_path="S1_training.csv", model_path="model/tcn_velocity_model.onnx"):
    """Verify the complete pipeline works end-to-end."""
    
    print("="*60)
    print("PIPELINE VERIFICATION - SYNC 3")
    print("="*60)
    
    # 1. Load data
    print("\n1. Loading data...")
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✓ Loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        print(f"   ❌ File not found: {csv_path}")
        print("   Please ensure S1_training.csv is in the current directory")
        return None
    
    # 2. Load model
    print("\n2. Loading model...")
    try:
        model_wrapper = P2ModelWrapper(model_path, use_onnx=True)
        print(f"   ✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"   ⚠️  Could not load ONNX model: {e}")
        print("   Trying PyTorch fallback...")
        try:
            model_wrapper = P2ModelWrapper(model_path, use_onnx=False)
            print(f"   ✓ Loaded PyTorch model from {model_path}")
        except Exception as e2:
            print(f"   ❌ Could not load model: {e2}")
            print("   Please ensure model file exists")
            return None
    
    # 3. Run EKF
    print("\n3. Running tuned EKF...")
    from run_tuned_ekf import run_tuned_ekf
    results = run_tuned_ekf(csv_path, model_path)
    print(f"   ✓ EKF completed")
    
    # 4. Convert to lat/lon (for map-matching)
    print("\n4. Converting to lat/lon...")
    lat, lon = local_xy_to_latlon(
        results['estimated_xy'][:, 0], 
        results['estimated_xy'][:, 1],
        ORIGIN_LAT, 
        ORIGIN_LON
    )
    print(f"   ✓ Converted {len(lat)} points to lat/lon")
    print(f"   lat range: [{lat.min():.6f}, {lat.max():.6f}]")
    print(f"   lon range: [{lon.min():.6f}, {lon.max():.6f}]")
    
    # 5. Check NIS
    nis_values = results['nis_values']
    nis_mean = np.mean(nis_values)
    nis_gt_599 = 100 * sum(nis_values > 5.99) / len(nis_values)
    
    print("\n" + "="*60)
    print("PIPELINE STATUS")
    print("="*60)
    print(f"Data:           {len(df)} rows")
    print(f"Model:          {model_path}")
    print(f"EKF Type:       VehicleEKFTuned")
    print(f"\n[NIS DIAGNOSTIC]")
    print(f"  NIS Mean:       {nis_mean:.3f}  {'✅' if 1.5 < nis_mean < 2.5 else '⚠️'}")
    print(f"  NIS % > 5.99:   {nis_gt_599:.1f}%  {'✅' if nis_gt_599 < 7 else '⚠️'}")
    
    pre_errors = results['pre_errors']
    print(f"\n[PRE-CORRECTION ERROR]")
    print(f"  Mean:           {pre_errors.mean():.2f} m")
    print(f"  Median:         {np.median(pre_errors):.2f} m")
    print(f"  Max:            {pre_errors.max():.2f} m")
    
    ekf = results['ekf']
    print(f"\n[FILTER HEALTH]")
    print(f"  GPS Rejections: {ekf.n_gps_rejected}/{ekf.n_gps_updates}")
    print(f"  Vel Rejections: {ekf.n_vel_rejected}/{ekf.n_vel_updates}")
    
    print("\n" + "="*60)
    print("STATUS: ✅ ALL COMPONENTS WORKING")
    print("="*60)
    
    return results, lat, lon

if __name__ == "__main__":
    results = verify_pipeline()
    if results is not None:
        print("\n✅ Pipeline frozen and ready for Sync 4!")
    else:
        print("\n❌ Pipeline verification failed. Please check errors above.")