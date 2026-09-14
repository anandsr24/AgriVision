"""
Detector module for AgriVision.

Wraps Ultralytics RT-DETR-L model into a standalone AgriculturalDetector class
that converts raw predictions into clean, structured JSON schemas.
"""

import io
import logging
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from ultralytics import RTDETR

logger = logging.getLogger("detector")

CLASS_MAP = {0: "crop", 1: "weed"}


class AgriculturalDetector:
    """
    Agricultural object detector for crop and weed detection using RT-DETR-L.
    """

    def __init__(
        self,
        model_path: str = "models/best.pt",
        confidence_threshold: float = 0.50,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path).resolve()
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.model: Optional[RTDETR] = None
        self._load_model()

    def _load_model(self):
        """Loads RT-DETR checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned model checkpoint not found at {self.model_path}"
            )

        self.model = RTDETR(str(self.model_path))

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
        conf_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Runs object detection on input image and returns structured predictions.

        Returns:
            Dict matching the specification:
            {
                "detections": [...],
                "counts": {"crop": int, "weed": int},
                "total_count": int,
                "inference_time_ms": float
            }
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        threshold = self.confidence_threshold if conf_threshold is None else float(conf_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        # Process image input
        if isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        t0 = time.perf_counter()
        # Run inference
                # Run inference
        gc.collect()
        results = self.model.predict(
            source=image,
            conf=threshold,
            device="cpu",
            imgsz=320,
            max_det=20,
            verbose=False,
        )
        inference_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        detections: List[Dict] = []
        counts = {"crop": 0, "weed": 0}

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(xyxy, confs, cls_ids):
                    conf_val = float(conf)
                    if conf_val < threshold:
                        continue

                    if cls_id not in CLASS_MAP:
                        logger.warning(f"Unknown class ID: {cls_id}")
                        continue

                    cls_name = CLASS_MAP[cls_id]
                    # Convert bbox to int pixel coordinates [x1, y1, x2, y2]
                    bbox_int = [int(round(coord)) for coord in box]

                    detections.append({
                        "class_id": int(cls_id),
                        "class": cls_name,
                        "confidence": round(conf_val, 4),
                        "bbox": bbox_int,
                    })

                    if cls_name in counts:
                        counts[cls_name] += 1
                    else:
                        counts[cls_name] = 1

        total_count = sum(counts.values())

        return {
            "detections": detections,
            "counts": counts,
            "total_count": total_count,
            "inference_time_ms": inference_time_ms,
        }
