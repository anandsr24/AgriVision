"""
FastAPI application for AgriVision.

Exposes REST endpoints for RT-DETR-L object detection and confidence-guarded reasoning over agricultural imagery.
"""

import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from src.detector import AgriculturalDetector
from src.reasoning import Intent, ReasoningEngine, classify_intent

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agrivision_api")

# Global instances
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.50"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
DEVICE = os.getenv("DEVICE", "cpu")

detector: Optional[AgriculturalDetector] = None
reasoning_engine: Optional[ReasoningEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes detector and reasoning engine on application startup."""
    global detector, reasoning_engine
    logger.info("Initializing AgriVision services...")

    try:
        model_file = Path(MODEL_PATH)
        if not model_file.exists():
            fallback = Path("rtdetr-l.pt")
            if fallback.exists():
                logger.warning(f"Primary model {model_file} not found; falling back to {fallback}")
                detector = AgriculturalDetector(
                    model_path=str(fallback),
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    device=DEVICE,
                )
            else:
                logger.error(f"No model checkpoint found at {model_file} or rtdetr-l.pt")
                detector = None
        else:
            detector = AgriculturalDetector(
                model_path=str(model_file),
                confidence_threshold=CONFIDENCE_THRESHOLD,
                device=DEVICE,
            )
            logger.info(f"Loaded detector model from {model_file}")

    except Exception as e:
        logger.error(f"Failed to load detector during startup: {e}")
        detector = None

    reasoning_engine = ReasoningEngine(confidence_threshold=CONFIDENCE_THRESHOLD)
    logger.info(f"ReasoningEngine initialized with confidence threshold: {CONFIDENCE_THRESHOLD}")
    yield


app = FastAPI(
    title="AgriVision API",
    description="RT-DETR Crop & Weed Detection and Reasoning API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_detector() -> AgriculturalDetector:
    """Helper to retrieve initialized detector or raise 503."""
    global detector
    if detector is None or detector.model is None:
        logger.error("Detector requested but model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently unavailable or could not be loaded.",
        )
    return detector


def validate_image_file(file: UploadFile, image_bytes: bytes) -> Image.Image:
    """Validates uploaded image format and integrity."""
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
        # Re-open after verify() as verify mutates file pointer
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil_img.size
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")
        return pil_img
    except (UnidentifiedImageError, ValueError, Exception) as e:
        logger.warning(f"Rejected invalid/corrupted image file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image format: {file.filename}",
        )


@app.get("/")
def get_root():
    """Returns basic API metadata."""
    return {
        "name": "AgriVision",
        "version": "1.0.0",
        "description": "RT-DETR crop and weed detection with deterministic reasoning",
    }


@app.get("/health")
def get_health():
    """Returns system health, model readiness, and configuration."""
    global detector
    is_loaded = detector is not None and detector.model is not None

    response_data = {
        "status": "ok" if is_loaded else "degraded",
        "model": "RT-DETR-L",
        "model_loaded": is_loaded,
        "classes": ["crop", "weed"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }

    if not is_loaded:
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=response_data)
    return response_data


@app.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    """
    Accepts an agricultural image and returns structured object detections for crops and weeds.
    """
    logger.info(f"POST /detect received file: {image.filename}")
    image_bytes = await image.read()
    pil_image = validate_image_file(image, image_bytes)
    # Limit image size for low-memory deployment
    max_size = 640
    if max(pil_image.size) > max_size:
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    det = get_detector()
    try:
        results = det.predict(pil_image)
        logger.info(
            f"Detection completed: total={results['total_count']}, "
            f"crops={results['counts']['crop']}, weeds={results['counts']['weed']} "
            f"({results['inference_time_ms']}ms)"
        )
        return results
    except Exception as e:
        logger.error(f"Inference error during /detect: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while running the object detector.",
        )


@app.post("/reason")
async def reason_query(
    image: UploadFile = File(...),
    question: str = Form(...),
):
    """
    Accepts an agricultural image and a natural language question, performing confidence-guarded reasoning.
    """
    cleaned_question = question.strip()
    logger.info(f"POST /reason received question: '{cleaned_question}' with image: {image.filename}")

    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question field cannot be empty.",
        )

    # 1. Classify intent before running model if possible
    intent = classify_intent(cleaned_question)
    engine = reasoning_engine or ReasoningEngine(confidence_threshold=CONFIDENCE_THRESHOLD)

    # If question is completely unsupported / out of scope, return immediately
    if intent == Intent.UNSUPPORTED:
        logger.info(f"Question '{cleaned_question}' classified as UNSUPPORTED. Returning guardrail answer.")
        return engine.reason(cleaned_question, detection_result=None)

    # 2. For supported questions, validate image and run detector
    image_bytes = await image.read()
    pil_image = validate_image_file(image, image_bytes)
    # Limit image size for low-memory deployment
    max_size = 640
    if max(pil_image.size) > max_size:
        pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    det = get_detector()
    try:
        detection_result = det.predict(pil_image)
    except Exception as e:
        logger.error(f"Inference error during /reason: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the image for reasoning.",
        )

    # 3. Perform deterministic reasoning
    reasoning_result = engine.reason(cleaned_question, detection_result=detection_result)
    logger.info(f"Reasoning answer: {reasoning_result['answer']} (confidence: {reasoning_result['confidence']})")
    return reasoning_result
