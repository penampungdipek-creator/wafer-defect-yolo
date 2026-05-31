#!/usr/bin/env python3
"""Export ONNX model to MIGraphX binary format (.mxr)."""

import argparse
from src.inference.migraphx_engine import MIGraphXEngine


def export(onnx_path: str, output: str = None, fp16: bool = True):
    """Compile ONNX to MIGraphX."""
    if output is None:
        output = onnx_path.replace(".onnx", ".mxr")
    
    MIGraphXEngine.export_onnx_to_migraphx(onnx_path, output, fp16=fp16)
    print(f"Exported to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to .onnx model")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()
    export(args.onnx, args.output, not args.no_fp16)
