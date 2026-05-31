"""Pydantic models for FastAPI request/response schemas."""

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    """Single detection result."""
    class_id: int = Field(..., description="Defect class ID")
    class_name: str = Field(..., description="Defect class name")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    bbox: list[float] = Field(..., description="[x1, y1, x2, y2] coordinates")
    severity: str = Field(..., description="Defect severity level")


class PredictionResponse(BaseModel):
    """Prediction API response."""
    detections: list[DetectionBox] = Field(default_factory=list)
    inference_time_ms: float = Field(..., description="Inference latency")
    image_size: list[int] = Field(..., description="[height, width]")
    defect_count: int = Field(..., description="Total defects detected")
    has_defects: bool = Field(..., description="Whether any defects found")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = True
    backend: str = ""
    uptime_seconds: float = 0.0


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str = ""
