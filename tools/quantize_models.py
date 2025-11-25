"""
ONNX Model Quantization Tool (FP32 → INT8)
==========================================

Mathematical Basis:
-------------------
FP32 weights: 4 bytes/parameter
INT8 weights: 1 byte/parameter
Compression: 4× size reduction

Quantization: w_int8 = round(w_fp32 / scale) * scale
- Introduces bounded error: |w_fp32 - w_int8| ≤ scale/2
- Error averages out across layers
- Expected accuracy drop: <1% on verification tasks

Performance Gains:
------------------
1. Memory bandwidth: 4× less data movement → ~2-4× speedup on memory-bound ops
2. INT8 SIMD: AVX2/AVX512 can compute more INT8 MACs per cycle
3. GPU Tensor Cores: INT8 support for massive parallelism

Expected Results:
-----------------
- RetinaFace (16MB → 4MB): 4× smaller, 2-3× faster detection
- ArcFace (166MB → 42MB): 4× smaller, 2-3× faster embedding

Usage:
------
python tools/quantize_models.py
"""

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import os


def get_model_size_mb(model_path):
    """Get model file size in MB."""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)


def quantize_model(input_path, output_path):
    """
    Quantize ONNX model from FP32 to INT8.
    
    Args:
        input_path: Path to FP32 ONNX model
        output_path: Path to save INT8 ONNX model
    """
    print(f"\n{'='*80}")
    print(f"Quantizing: {input_path.name}")
    print(f"{'='*80}")
    
    # Check if input exists
    if not input_path.exists():
        print(f"❌ Error: Model not found at {input_path}")
        return False
    
    # Get original size
    original_size = get_model_size_mb(input_path)
    print(f"Original model size: {original_size:.2f} MB")
    
    try:
        # Quantize dynamically (no calibration data needed)
        # This converts weights from FP32 to INT8
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8,  # Unsigned INT8
            per_channel=True,  # Better accuracy with per-channel quantization
        )
        
        # Get quantized size
        quantized_size = get_model_size_mb(output_path)
        compression_ratio = original_size / quantized_size
        size_reduction_pct = (1 - quantized_size / original_size) * 100
        
        print(f"✅ Quantization successful!")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}× (saved {size_reduction_pct:.1f}%)")
        print(f"Saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return False


def main():
    """Quantize all models."""
    print("\n" + "="*80)
    print("ONNX MODEL QUANTIZATION (FP32 → INT8)")
    print("="*80)
    print("\nMathematical Basis:")
    print("  - FP32: 4 bytes/param")
    print("  - INT8: 1 byte/param")
    print("  - Compression: 4× size reduction")
    print("  - Expected speedup: 2-3× per model")
    print("  - Accuracy loss: <1% (negligible)")
    
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    # Models to quantize
    models = [
        {
            'name': 'RetinaFace',
            'input': models_dir / 'retinaface_resnet50.onnx',
            'output': models_dir / 'retinaface_resnet50_int8.onnx'
        },
        {
            'name': 'ArcFace',
            'input': models_dir / 'arcface_resnet100.onnx',
            'output': models_dir / 'arcface_resnet100_int8.onnx'
        }
    ]
    
    results = []
    
    for model in models:
        success = quantize_model(model['input'], model['output'])
        results.append({
            'name': model['name'],
            'success': success,
            'output': model['output']
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("QUANTIZATION SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{result['name']:20s} {status}")
        if result['success']:
            print(f"  → {result['output']}")
    
    # Instructions
    successful_count = sum(1 for r in results if r['success'])
    if successful_count > 0:
        print(f"\n{'-'*80}")
        print("NEXT STEPS:")
        print(f"{'-'*80}")
        print("Update config files to use INT8 models:")
        print("  1. config/detector_config.yaml:")
        print("       model_path: models/retinaface_resnet50_int8.onnx")
        print("  2. config/embedder_config.yaml:")
        print("       model_path: models/arcface_resnet100_int8.onnx")
        print(f"\nExpected combined speedup: 2-3× (per model)")
        print(f"Expected combined compression: ~4× smaller")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
