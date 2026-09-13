"""
API integration and unit tests for AgriVision FastAPI service.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src import main
from src.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes():
    """Creates a simple valid JPEG image in memory for testing."""
    img = Image.new("RGB", (640, 640), color=(100, 200, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ============================================================
# ENDPOINT TESTS: GET / and GET /health
# ============================================================

def test_get_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AgriVision"
    assert data["version"] == "1.0.0"
    assert "RT-DETR" in data["description"]


def test_get_health(client):
    response = client.get("/health")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert data["classes"] == ["crop", "weed"]


def test_get_health_model_unavailable(client, monkeypatch):
    monkeypatch.setattr(main, "detector", None)
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False


# ============================================================
# ENDPOINT TESTS: POST /detect
# ============================================================

def test_post_detect_valid_image(client, sample_image_bytes, monkeypatch):
    # Mock detector to test API schema deterministically
    mock_det = MagicMock()
    mock_det.model = MagicMock()
    mock_det.predict.return_value = {
        "detections": [
            {"class_id": 0, "class": "crop", "confidence": 0.92, "bbox": [10, 20, 100, 200]},
            {"class_id": 1, "class": "weed", "confidence": 0.85, "bbox": [150, 120, 250, 280]},
        ],
        "counts": {"crop": 1, "weed": 1},
        "total_count": 2,
        "inference_time_ms": 12.5,
    }
    monkeypatch.setattr(main, "detector", mock_det)

    response = client.post(
        "/detect",
        files={"image": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["detections"]) == 2
    assert data["counts"]["crop"] == 1
    assert data["counts"]["weed"] == 1
    assert data["total_count"] == 2
    assert data["inference_time_ms"] == 12.5


def test_post_detect_invalid_image_format(client):
    response = client.post(
        "/detect",
        files={"image": ("bad.txt", b"not an image file data", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "Invalid or corrupted image format" in data["detail"]


def test_post_detect_empty_file(client):
    response = client.post(
        "/detect",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"]


def test_post_detect_service_unavailable_when_no_model(client, sample_image_bytes, monkeypatch):
    monkeypatch.setattr(main, "detector", None)
    response = client.post(
        "/detect",
        files={"image": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


# ============================================================
# ENDPOINT TESTS: POST /reason
# ============================================================

def test_post_reason_supported_query(client, sample_image_bytes, monkeypatch):
    mock_det = MagicMock()
    mock_det.model = MagicMock()
    mock_det.predict.return_value = {
        "detections": [
            {"class_id": 1, "class": "weed", "confidence": 0.88, "bbox": [50, 50, 100, 100]},
            {"class_id": 1, "class": "weed", "confidence": 0.79, "bbox": [200, 200, 300, 300]},
        ],
        "counts": {"crop": 0, "weed": 2},
        "total_count": 2,
        "inference_time_ms": 14.1,
    }
    monkeypatch.setattr(main, "detector", mock_det)

    response = client.post(
        "/reason",
        files={"image": ("field.jpg", sample_image_bytes, "image/jpeg")},
        data={"question": "How many weeds are present?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "count"
    assert "2 weeds" in data["answer"]
    assert data["confidence"] == "high"
    assert data["evidence"]["weed_count"] == 2


def test_post_reason_unsupported_query(client, sample_image_bytes):
    response = client.post(
        "/reason",
        files={"image": ("field.jpg", sample_image_bytes, "image/jpeg")},
        data={"question": "What fertilizer should I apply to these crops?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "unsupported"
    assert "Insufficient information" in data["answer"]
    assert data["confidence"] == "insufficient"


def test_post_reason_empty_question(client, sample_image_bytes):
    response = client.post(
        "/reason",
        files={"image": ("field.jpg", sample_image_bytes, "image/jpeg")},
        data={"question": "   "},
    )
    assert response.status_code == 400
    data = response.json()
    assert "cannot be empty" in data["detail"]


# ============================================================
# REAL END-TO-END VERIFICATION (without mocks)
# ============================================================

def test_real_end_to_end_detect_and_reason(client, sample_image_bytes):
    """Verifies unmocked inference through real detector if checkpoint is loaded."""
    if main.detector is None or main.detector.model is None:
        pytest.skip("Real model checkpoint not loaded in main app context")

    # 1. Real /detect
    res_detect = client.post(
        "/detect",
        files={"image": ("field.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert res_detect.status_code == 200
    detect_data = res_detect.json()
    assert "detections" in detect_data
    assert "counts" in detect_data
    assert "total_count" in detect_data
    assert detect_data["total_count"] == detect_data["counts"]["crop"] + detect_data["counts"]["weed"]

    # 2. Real /reason
    res_reason = client.post(
        "/reason",
        files={"image": ("field.jpg", sample_image_bytes, "image/jpeg")},
        data={"question": "What objects are present in the image?"},
    )
    assert res_reason.status_code == 200
    reason_data = res_reason.json()
    assert reason_data["intent"] == "summary"
    assert "answer" in reason_data
    assert reason_data["confidence"] in ("high", "insufficient")
