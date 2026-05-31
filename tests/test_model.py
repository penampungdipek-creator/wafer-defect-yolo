"""Tests for model inference."""

import numpy as np
import pytest


def test_detection_dataclass():
    """Test Detection dataclass properties."""
    from src.models.yolo_rocm import Detection

    det = Detection(
        class_id=0,
        class_name="scratch",
        confidence=0.95,
        bbox=(10.0, 20.0, 50.0, 80.0),
        defect_severity="medium",
    )
    assert det.area == 40.0 * 60.0
    assert det.class_name == "scratch"


def test_inference_result():
    """Test InferenceResult aggregation."""
    from src.models.yolo_rocm import InferenceResult, Detection

    det1 = Detection(0, "scratch", 0.9, (0, 0, 10, 10), "medium")
    det2 = Detection(1, "particle", 0.8, (20, 20, 30, 30), "high")

    result = InferenceResult(
        image_path="test.png",
        detections=[det1, det2],
        inference_time_ms=5.0,
    )
    assert result.defect_count == 2
    assert result.has_defects is True
    assert result.total_time_ms == 5.0


def test_severity_mapping():
    """Test all defect classes have severity mapping."""
    from src.models.yolo_rocm import CLASS_NAMES, SEVERITY_MAP

    for cls in CLASS_NAMES:
        assert cls in SEVERITY_MAP, f"Missing severity for {cls}"
        assert SEVERITY_MAP[cls] in ("low", "medium", "high", "critical")
