#!/usr/bin/env python3
"""Export YOLOv8 model to ONNX format."""

import argparse
from pathlib import Path
from ultralytics import YOLO


def export(weights: str, img_size: int = 640, dynamic: bool = True, simplify: bool = True):
    """Export YOLO to ONNX.
    
    Args:
        weights: Path to .pt weights.
        img_size: Input image size.
        dynamic: Enable dynamic batch dimension.
        simplify: Run ONNX simplifier.
    """
    model = YOLO(weights)
    
    output = Path(weights).with_suffix(".onnx")
    model.export(
        format="onnx",
        imgsz=img_size,
        dynamic=dynamic,
        simplify=simplify,
    )
    print(f"Exported to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--no-dynamic", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()
    export(args.weights, args.img_size, not args.no_dynamic, not args.no_simplify)
