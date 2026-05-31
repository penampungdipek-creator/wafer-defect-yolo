"""FastAPI REST API for wafer defect detection.

Provides /predict, /health, and /metrics endpoints
with automatic Prometheus instrumentation.
"""

import io
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse

from .schemas import PredictionResponse, DetectionBox, HealthResponse
from ..models.yolo_rocm import WaferDefectYOLO, CLASS_NAMES
from ..monitoring.metrics import (
    INFERENCE_LATENCY, INFERENCE_TOTAL, DEFECT_DISTRIBUTION, ACTIVE_REQUESTS
)

logger = logging.getLogger(__name__)

model: WaferDefectYOLO | None = None
start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    weights_path = Path("weights/wafer_yolov8l.pt")
    if weights_path.exists():
        model = WaferDefectYOLO(str(weights_path))
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Weights not found at {weights_path}")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Wafer Defect YOLO API",
    description="Semiconductor wafer defect detection using YOLOv8 with ROCm optimization",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        backend=model.backend if model else "none",
        uptime_seconds=time.time() - start_time,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, ge=0, le=1),
    iou_threshold: float = Query(0.45, ge=0, le=1),
):
    """Run defect detection on uploaded wafer image.
    
    Args:
        file: Image file (PNG, JPEG, TIFF).
        conf_threshold: Confidence threshold for detections.
        iou_threshold: IoU threshold for NMS.
        
    Returns:
        PredictionResponse with detections and metadata.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    ACTIVE_REQUESTS.inc()
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Run inference
        with INFERENCE_LATENCY.time():
            result = model.predict(img, conf=conf_threshold, iou=iou_threshold)

        # Track metrics
        INFERENCE_TOTAL.inc()
        for det in result.detections:
            DEFECT_DISTRIBUTION.labels(defect_class=det.class_name).inc()

        # Format response
        detections = [
            DetectionBox(
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=round(det.confidence, 4),
                bbox=[round(x, 2) for x in det.bbox],
                severity=det.defect_severity,
            )
            for det in result.detections
        ]

        return PredictionResponse(
            detections=detections,
            inference_time_ms=round(result.inference_time_ms, 2),
            image_size=list(img.shape[:2]),
            defect_count=result.defect_count,
            has_defects=result.has_defects,
        )
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/classes")
async def get_classes():
    """Get supported defect classes."""
    return {"classes": CLASS_NAMES, "count": len(CLASS_NAMES)}
