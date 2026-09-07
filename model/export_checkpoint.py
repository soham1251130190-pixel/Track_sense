"""
export_checkpoint.py
Exports trained TCN PyTorch model to ONNX format for mobile / embedded deployment.

Features:
1. Loads best trained weights (best_model_real.pt)
2. Exports ONNX model with dynamic batch axis
3. Validates graph with onnx.checker
4. Verifies numerical parity using ONNX Runtime
5. Benchmarks latency per inference window (ms)
"""
import os
import sys
import time
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import torch
import onnx
import onnxruntime as ort

from model import build_model, count_parameters

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def export_tcn_to_onnx(checkpoint_path=None, output_onnx_path=None):
    """
    Export PyTorch model to ONNX.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        output_onnx_path: Path to save .onnx model
    """
    print("=" * 60)
    print("EXPORT FIRST CHECKPOINT TO ONNX (HOURS 6-10)")
    print("=" * 60)
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(SCRIPT_DIR, 'best_model_real.pt')
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(SCRIPT_DIR, 'best_model_synthetic.pt')
            
    if output_onnx_path is None:
        output_onnx_path = os.path.join(SCRIPT_DIR, 'tcn_velocity_model.onnx')
        
    print(f"📦 Loading weights from: {checkpoint_path}")
    model = build_model()
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')} (Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
        else:
            model.load_state_dict(checkpoint)
            print("   ✅ Loaded model weights successfully")
    else:
        print("   ⚠️  Checkpoint not found, exporting initialized architecture")
        
    model.eval()
    
    # 1. Prepare Dummy Input: (batch_size=1, in_channels=6, window_size=100)
    batch_size = 1
    in_channels = 6
    window_size = 100
    dummy_input = torch.randn(batch_size, in_channels, window_size, dtype=torch.float32)
    
    # 2. Export to ONNX
    print(f"\n⚙️  Exporting to ONNX: {output_onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=18,
        input_names=['imu_input'],
        output_names=['velocity_outputs'],
        dynamic_axes={
            'imu_input': {0: 'batch_size'},
            'velocity_outputs': {0: 'batch_size'}
        }
    )
    print(f"   ✅ Export complete!")
    
    # 3. Validate ONNX Model
    print("\n🔍 Validating ONNX Graph Structure...")
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("   ✅ ONNX model is structurally valid!")
    
    # Check file size
    file_size_kb = os.path.getsize(output_onnx_path) / 1024
    print(f"   📁 ONNX Model Size: {file_size_kb:.1f} KB")
    
    # 4. Numerical Parity Check with ONNX Runtime
    print("\n🧪 Testing Inference & Numerical Parity with ONNX Runtime...")
    ort_session = ort.InferenceSession(output_onnx_path, providers=['CPUExecutionProvider'])
    
    test_input = np.random.randn(4, 6, 100).astype(np.float32)
    
    # PyTorch output
    with torch.no_grad():
        torch_output = model(torch.from_numpy(test_input)).numpy()
        
    # ONNX Runtime output
    ort_inputs = {ort_session.get_inputs()[0].name: test_input}
    ort_outputs = ort_session.run(None, ort_inputs)[0]
    
    # Difference check
    max_diff = np.max(np.abs(torch_output - ort_outputs))
    print(f"   PyTorch Output Shape: {torch_output.shape}")
    print(f"   ONNX Output Shape:    {ort_outputs.shape}")
    print(f"   Maximum Absolute Difference: {max_diff:.2e}")
    
    assert max_diff < 1e-4, f"Parity check failed! Diff = {max_diff}"
    print("   ✅ PyTorch and ONNX Runtime outputs match within tolerance (< 1e-4)!")
    
    # 5. Latency Benchmarking (100 runs)
    print("\n⏱️  Benchmarking Inference Latency (CPU)...")
    single_input = np.random.randn(1, 6, 100).astype(np.float32)
    latencies = []
    
    # Warmup
    for _ in range(10):
        ort_session.run(None, {ort_session.get_inputs()[0].name: single_input})
        
    for _ in range(100):
        t0 = time.perf_counter()
        ort_session.run(None, {ort_session.get_inputs()[0].name: single_input})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
        
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"   Mean Latency: {avg_latency:.2f} ms per window")
    print(f"   95th Percentile: {p95_latency:.2f} ms")
    print(f"   Max Real-Time Throughput: {1000 / avg_latency:.0f} Hz (Requirement: 10 Hz)")
    print(f"   ✅ Real-time margin: {(1000 / avg_latency) / 10:.1f}x faster than real-time!")
    
    return {
        'onnx_path': output_onnx_path,
        'model_size_kb': file_size_kb,
        'max_diff': float(max_diff),
        'mean_latency_ms': float(avg_latency),
        'p95_latency_ms': float(p95_latency),
        'success': True
    }


if __name__ == "__main__":
    export_tcn_to_onnx()

