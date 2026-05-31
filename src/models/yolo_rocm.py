"""ROCm-optimized YOLO wrapper for wafer defect detection.

Supports both NVIDIA CUDA and AMD ROCm backends with automatic
environment configuration for optimal GPU utilization.
"""

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single defect detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    defect_severity: str = "medium"

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class InferenceResult:
    """Complete inference result for a single image."""
    image_path: str
    detections: list[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0

    @property
    def total_time_ms(self) -> float:
        return self.inference_time_ms + self.preprocessing_time_ms + self.postprocessing_time_ms

    @property
    def defect_count(self) -> int:
        return len(self.detections)

    @property
    def has_defects(self) -> bool:
        return len(self.detections) > 0


CLASS_NAMES = [
    "scratch", "particle", "edge_chip", "void", "pattern_shift",
    "bridge", "missing_bond", "crack", "contamination", "delamination"
]

SEVERITY_MAP = {
    "scratch": "medium", "particle": "high", "edge_chip": "medium",
    "void": "high", "pattern_shift": "critical", "bridge": "critical",
    "missing_bond": "high", "crack": "critical", "contamination": "medium",
    "delamination": "high"
}


def setup_rocm_environment(gpu_arch: str = "gfx1100") -> None:
    """Configure ROCm environment variables for optimal performance.
    
    Args:
        gpu_arch: GPU architecture string (e.g., gfx1100 for RX 7900 XTX).
    """
    env_vars = {
        "HSA_OVERRIDE_GFX_VERSION": gpu_arch.replace("gfx", "").replace("1100", "11.0.0"),
        "HSA_ENABLE_SDMA": "0",
        "PYTORCH_HIP_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:512",
        "HIP_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "0",
        "GPU_MAX_HW_QUEUES": "8",
    }
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info(f"Set {key}={value}")


def detect_gpu_backend() -> tuple[str, str]:
    """Auto-detect available GPU backend.
    
    Returns:
        Tuple of (backend_name, device_string).
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        if hasattr(torch.version, "hip") and torch.version.hip:
            return "rocm", f"rocm:0 ({device_name})"
        return "cuda", f"cuda:0 ({device_name})"
    return "cpu", "cpu"


class WaferDefectYOLO:
    """High-level YOLO wrapper with ROCm optimization.
    
    Provides a clean interface for wafer defect detection with
    automatic backend selection, environment configuration, and
    structured inference results.
    
    Example:
        >>> model = WaferDefectYOLO("weights/wafer_yolov8l.pt")
        >>> results = model.predict("wafer_image.png")
        >>> for det in results.detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f}")
    """

    SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def __init__(
        self,
        weights: str,
        device: Optional[str] = None,
        gpu_arch: str = "gfx1100",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
    ):
        self.weights = Path(weights)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        # Auto-detect backend
        if device is None:
            backend, self.device = detect_gpu_backend()
        else:
            self.device = device
            backend = "cuda" if "cuda" in device else "rocm" if "rocm" in device else "cpu"

        # Setup ROCm environment if needed
        if backend == "rocm":
            setup_rocm_environment(gpu_arch)

        logger.info(f"Loading model {weights} on {self.device}")
        self.model = YOLO(str(weights))
        self.backend = backend

        # Warmup
        self._warmup()

    def _warmup(self, n: int = 3) -> None:
        """Warmup GPU with dummy inference."""
        dummy = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        for _ in range(n):
            self.model.predict(dummy, verbose=False)
        logger.info(f"Warmup complete ({n} iterations)")

    def predict(
        self,
        image: str | np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> InferenceResult:
        """Run inference on a single image.
        
        Args:
            image: Image path or numpy array (H, W, 3).
            conf: Confidence threshold override.
            iou: IoU threshold override.
            
        Returns:
            InferenceResult with detections and timing info.
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold

        t0 = time.perf_counter()
        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            imgsz=self.img_size,
            verbose=False,
        )
        inference_time = (time.perf_counter() - t0) * 1000

        result = results[0]
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append(Detection(
                class_id=cls_id,
                class_name=CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}",
                confidence=float(box.conf[0]),
                bbox=tuple(box.xyxy[0].cpu().numpy().tolist()),
                defect_severity=SEVERITY_MAP.get(CLASS_NAMES[cls_id], "medium"),
            ))

        image_path = str(image) if isinstance(image, (str, Path)) else "<array>"
        return InferenceResult(
            image_path=image_path,
            detections=detections,
            inference_time_ms=inference_time,
        )

    def predict_batch(
        self,
        images: list[str | np.ndarray],
        batch_size: int = 8,
    ) -> list[InferenceResult]:
        """Run inference on a batch of images.
        
        Args:
            images: List of image paths or numpy arrays.
            batch_size: Batch size for GPU utilization.
            
        Returns:
            List of InferenceResult.
        """
        results = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            for img in batch:
                results.append(self.predict(img))
        return results

    def get_critical_defects(self, result: InferenceResult) -> list[Detection]:
        """Filter for critical/high severity defects only."""
        return [d for d in result.detections if d.defect_severity in ("critical", "high")]

    def summary(self, result: InferenceResult) -> str:
        """Human-readable summary of inference result."""
        lines = [
            f"Image: {result.image_path}",
            f"Defects found: {result.defect_count}",
            f"Inference: {result.inference_time_ms:.1f}ms",
            f"Backend: {self.backend}",
        ]
        for det in result.detections:
            lines.append(f"  [{det.defect_severity.upper()}] {det.class_name}: {det.confidence:.2%}")
        return "\n".join(lines)
