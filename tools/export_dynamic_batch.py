"""
Re-export the ArcFace ONNX model with a dynamic (symbolic) batch axis.

Why
---
A static input shape [1, 3, 112, 112] forces one ONNX call per face. A symbolic
batch dim ['N', 3, 112, 112] lets the embedder stack many faces into one call:
batch inference scales sublinearly (batch_8 ≈ 3× batch_1, not 8×).

This is a one-time tool. It only needs the `onnx` package (not a runtime dep).

Usage
-----
    python tools/export_dynamic_batch.py
    python tools/export_dynamic_batch.py --input models/arcface_resnet100.onnx \
                                         --output models/arcface_resnet100_dynamic.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx


def make_dynamic(input_path: str, output_path: str, dim_name: str = "N") -> None:
    model = onnx.load(input_path)

    # Set first dim of the primary input + output to a symbolic name.
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = dim_name
    model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = dim_name

    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"[OK] Saved dynamic-batch model -> {output_path}")
    print(f"     input '{model.graph.input[0].name}' batch dim = '{dim_name}'")


def validate(output_path: str) -> bool:
    """Confirm batch sizes 1, 4, 8 all run and produce [N, 512]."""
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"  input shape reported by ORT: {sess.get_inputs()[0].shape}")

    ok = True
    for n in (1, 4, 8):
        dummy = np.random.randn(n, 3, 112, 112).astype(np.float32)
        out = sess.run([out_name], {in_name: dummy})[0]
        status = "OK" if out.shape[0] == n and out.shape[1] == 512 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  batch={n}: output shape {out.shape} [{status}]")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Export ArcFace ONNX with dynamic batch axis")
    ap.add_argument("--input", default="models/arcface_resnet100.onnx")
    ap.add_argument("--output", default="models/arcface_resnet100_dynamic.onnx")
    ap.add_argument("--dim-name", default="N")
    args = ap.parse_args()

    if not Path(args.input).exists():
        raise SystemExit(f"Input model not found: {args.input}")

    make_dynamic(args.input, args.output, args.dim_name)

    print("\nValidating with onnxruntime (batch sizes 1, 4, 8)...")
    if validate(args.output):
        print("\n[OK] Validation passed. Point embedder_config.yaml model.path at the dynamic model.")
    else:
        raise SystemExit("[FAIL] Validation failed — output shapes did not scale with batch size.")


if __name__ == "__main__":
    main()
