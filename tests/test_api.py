"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from src.api.main import app
    return TestClient(app)


def test_health_endpoint(client):
    """Test /health returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_classes_endpoint(client):
    """Test /classes returns defect class list."""
    response = client.get("/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert data["count"] == 10


def test_predict_without_model(client):
    """Test /predict returns 503 when model not loaded."""
    import numpy as np
    import cv2

    # Create dummy image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)

    response = client.post(
        "/predict",
        files={"file": ("test.png", buf.tobytes(), "image/png")},
    )
    # Should return 503 if model not loaded, or 200 if loaded
    assert response.status_code in (200, 503)
