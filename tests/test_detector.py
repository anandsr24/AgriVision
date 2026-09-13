"""
Unit and integration tests for AgriculturalDetector module (src/detector.py).
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.detector import CLASS_MAP, AgriculturalDetector


@pytest.fixture
def sample_pil_image():
    """Generates a dummy RGB test image."""
    return Image.new("RGB", (640, 640), color=(120, 200, 100))


@pytest.fixture
def sample_image_bytes(sample_pil_image):
    """Generates JPEG bytes in memory."""
    import io
    buf = io.BytesIO()
    sample_pil_image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def mock_detector():
    """Creates an AgriculturalDetector with mocked RTDETR model for deterministic testing."""
    with patch("src.detector.RTDETR") as mock_rtdetr_cls:
        mock_model = MagicMock()
        mock_rtdetr_cls.return_value = mock_model

        detector = AgriculturalDetector(
            model_path="models/best.pt",
            confidence_threshold=0.50,
            device="cpu",
        )
        detector.model = mock_model
        return detector, mock_model


def make_mock_result(xyxy: np.ndarray, confs: np.ndarray, cls_ids: np.ndarray):
    """Creates a mock Ultralytics Results object containing Boxes."""
    mock_boxes = MagicMock()
    mock_boxes.__len__.return_value = len(xyxy)
    mock_boxes.xyxy.cpu().numpy.return_value = xyxy
    mock_boxes.conf.cpu().numpy.return_value = confs
    mock_boxes.cls.cpu().numpy.return_value = cls_ids

    mock_res = MagicMock()
    mock_res.boxes = mock_boxes
    return mock_res


# =============================================================================
# 1. Output Schema & JSON Compatibility Tests
# =============================================================================

def test_detector_output_schema_mocked(mock_detector, sample_pil_image):
    detector, mock_model = mock_detector

    xyxy = np.array([
        [10.2, 20.4, 100.8, 200.1],
        [150.0, 120.5, 250.2, 280.9],
    ])
    confs = np.array([0.92, 0.85])
    cls_ids = np.array([0, 1])

    mock_model.predict.return_value = [make_mock_result(xyxy, confs, cls_ids)]

    results = detector.predict(sample_pil_image)

    # 1. Verify schema keys
    assert "detections" in results
    assert "counts" in results
    assert "total_count" in results
    assert "inference_time_ms" in results

    # 2. Verify types
    assert isinstance(results["detections"], list)
    assert isinstance(results["counts"], dict)
    assert isinstance(results["total_count"], int)
    assert isinstance(results["inference_time_ms"], float)

    # 3. Verify JSON serialization
    json_str = json.dumps(results)
    assert isinstance(json_str, str)
    deserialized = json.loads(json_str)
    assert deserialized["total_count"] == 2


# =============================================================================
# 2. Class Names and Counts Consistency
# =============================================================================

def test_detector_classes_and_counts(mock_detector, sample_pil_image):
    detector, mock_model = mock_detector

    xyxy = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [60.0, 60.0, 120.0, 120.0],
        [200.0, 200.0, 300.0, 300.0],
    ])
    confs = np.array([0.95, 0.88, 0.72])
    cls_ids = np.array([0, 0, 1])  # 2 crops, 1 weed

    mock_model.predict.return_value = [make_mock_result(xyxy, confs, cls_ids)]

    results = detector.predict(sample_pil_image)

    assert results["counts"]["crop"] == 2
    assert results["counts"]["weed"] == 1
    assert results["total_count"] == 3
    assert len(results["detections"]) == 3

    for det in results["detections"]:
        assert det["class"] in ("crop", "weed")
        assert det["class_id"] in (0, 1)
        assert CLASS_MAP[det["class_id"]] == det["class"]


# =============================================================================
# 3. Confidence Filtering
# =============================================================================

def test_detector_confidence_filtering(mock_detector, sample_pil_image):
    detector, mock_model = mock_detector

    xyxy = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [60.0, 60.0, 120.0, 120.0],
    ])
    confs = np.array([0.90, 0.35])
    cls_ids = np.array([0, 1])

    mock_model.predict.return_value = [make_mock_result(xyxy, confs, cls_ids)]

    # Run with default threshold 0.50 -> 0.35 weed should be filtered
    results_50 = detector.predict(sample_pil_image, conf_threshold=0.50)
    assert results_50["total_count"] == 1
    assert results_50["counts"]["crop"] == 1
    assert results_50["counts"]["weed"] == 0
    assert results_50["detections"][0]["confidence"] >= 0.50

    # Run with custom low threshold 0.30 -> both should pass
    results_30 = detector.predict(sample_pil_image, conf_threshold=0.30)
    assert results_30["total_count"] == 2
    assert results_30["counts"]["crop"] == 1
    assert results_30["counts"]["weed"] == 1
def test_invalid_confidence_threshold(mock_detector, sample_pil_image):
    detector, _ = mock_detector

    with pytest.raises(ValueError, match="between 0 and 1"):
        detector.predict(sample_pil_image, conf_threshold=-0.1)

    with pytest.raises(ValueError, match="between 0 and 1"):
        detector.predict(sample_pil_image, conf_threshold=1.1)
def test_unknown_class_is_ignored(mock_detector, sample_pil_image):
    detector, mock_model = mock_detector

    xyxy = np.array([[10.0, 10.0, 50.0, 50.0]])
    confs = np.array([0.95])
    cls_ids = np.array([5])

    mock_model.predict.return_value = [
        make_mock_result(xyxy, confs, cls_ids)
    ]

    results = detector.predict(sample_pil_image)

    assert results["total_count"] == 0
    assert results["counts"] == {"crop": 0, "weed": 0}
    assert results["detections"] == []
# =============================================================================
# 4. Bounding Box Structure & Bounds
# =============================================================================

def test_detector_bbox_structure_and_bounds(mock_detector, sample_pil_image):
    detector, mock_model = mock_detector

    xyxy = np.array([
        [15.3, 25.7, 120.4, 250.9],
    ])
    confs = np.array([0.91])
    cls_ids = np.array([0])

    mock_model.predict.return_value = [make_mock_result(xyxy, confs, cls_ids)]

    results = detector.predict(sample_pil_image)
    assert len(results["detections"]) == 1
    det = results["detections"][0]
    bbox = det["bbox"]

    # Must be list of 4 ints
    assert len(bbox) == 4
    assert all(isinstance(coord, int) for coord in bbox)
    x1, y1, x2, y2 = bbox
    assert x1 == 15
    assert y1 == 26
    assert x2 == 120
    assert y2 == 251
    assert x1 <= x2
    assert y1 <= y2
    assert x1 >= 0 and y1 >= 0


# =============================================================================
# 5. Input Format Polymorphism
# =============================================================================

def test_detector_input_formats(mock_detector, sample_pil_image, sample_image_bytes, tmp_path):
    detector, mock_model = mock_detector

    xyxy = np.empty((0, 4))
    confs = np.empty((0,))
    cls_ids = np.empty((0,))
    mock_model.predict.return_value = [make_mock_result(xyxy, confs, cls_ids)]

    # 1. PIL Image
    res1 = detector.predict(sample_pil_image)
    assert res1["total_count"] == 0

    # 2. Bytes
    res2 = detector.predict(sample_image_bytes)
    assert res2["total_count"] == 0

    # 3. Numpy array
    np_arr = np.array(sample_pil_image)
    res3 = detector.predict(np_arr)
    assert res3["total_count"] == 0

    # 4. File Path (string & Path object)
    test_img_file = tmp_path / "test.jpg"
    sample_pil_image.save(test_img_file)
    res4 = detector.predict(str(test_img_file))
    assert res4["total_count"] == 0
    res5 = detector.predict(test_img_file)
    assert res5["total_count"] == 0


# =============================================================================
# 6. Real Checkpoint Integration Test (if models/best.pt exists)
# =============================================================================

def test_real_detector_with_checkpoint(sample_pil_image):
    model_path = Path("models/best.pt")
    if not model_path.exists():
        pytest.skip("models/best.pt not present for real integration test")

    detector = AgriculturalDetector(model_path=str(model_path), confidence_threshold=0.50)
    results = detector.predict(sample_pil_image)

    assert "detections" in results
    assert "counts" in results
    assert "total_count" in results
    assert "inference_time_ms" in results
    assert results["inference_time_ms"] > 0.0
    assert results["total_count"] == results["counts"]["crop"] + results["counts"]["weed"]
