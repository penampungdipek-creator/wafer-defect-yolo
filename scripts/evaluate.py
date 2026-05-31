#!/usr/bin/env python3
"""Evaluation and benchmarking script.

Usage:
    python scripts/evaluate.py --weights weights/wafer_yolov8l.pt --benchmark
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO
from src.inference.benchmark import MultiBackendBenchmark


def evaluate(weights: str, data: str = None, benchmark: bool = False):
    """Run evaluation and optional benchmark."""
    model = YOLO(weights)

    # Validation
    if data:
        results = model.val(data=data)
        print(f"\nmAP@50: {results.box.map50:.4f}")
        print(f"mAP@50-95: {results.box.map:.4f}")
        for i, name in enumerate(results.names.values()):
            print(f"  {name}: {results.box.ap50[i]:.4f}")

    # Benchmark
    if benchmark:
        bench = MultiBackendBenchmark(img_size=640, iterations=50)
        results = []

        # PyTorch (auto-detect backend)
        try:
            r = bench.benchmark_pytorch(weights)
            results.append(r)
            print(f"✓ {r.summary}")
        except Exception as e:
            print(f"✗ PyTorch: {e}")

        # Export and benchmark ONNX
        try:
            model.export(format="onnx", imgsz=640, dynamic=True)
            onnx_path = weights.replace(".pt", ".onnx")
            r = bench.benchmark_onnxruntime(onnx_path)
            results.append(r)
            print(f"✓ {r.summary}")
        except Exception as e:
            print(f"✗ ONNX: {e}")

        bench.print_comparison(results)
        bench.export_json(results, "benchmark_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    evaluate(args.weights, args.data, args.benchmark)
