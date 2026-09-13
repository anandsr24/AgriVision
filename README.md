# AgriVision — RT-DETR Crop & Weed Detection and Reasoning API

> **"An RT-DETR-powered agricultural vision API that detects crops and weeds and provides a lightweight reasoning layer for counting, comparison, and confidence-aware analysis."**

---

## 1. Problem Statement
In precision agriculture, automated weed management and crop monitoring require accurate, real-time spatial localization of crop plants and competing weed vegetation. Traditional computer vision approaches often fail under varying field illumination, complex soil backgrounds, and severe visual overlap between morphological plant features. AgriVision provides an end-to-end, reproducible solution by pairing a fine-tuned **Ultralytics RT-DETR-L** real-time transformer detector with a **deterministic, confidence-guarded reasoning layer** exposed via high-performance FastAPI endpoints.

---

## 2. Motivation
- **Targeted Herbicide & Mechanical Weeding**: Differentiating weeds from crops at the pixel/bounding-box level enables site-specific intervention, reducing chemical usage by up to 80%.
- **Explainable, Grounded Reasoning**: Rather than using hallucination-prone Large Language Models (LLMs) or complex agent frameworks for agronomic analysis, AgriVision relies on deterministic Python intent routing that reasons exclusively over structured, confidence-filtered detector outputs.
- **Strict Evidence Guardrails**: In agricultural operations, false certainty can destroy crops. AgriVision strictly refuses to guess—returning `"Insufficient information"` whenever detections lack confidence or user questions fall outside the detector's capability (such as disease diagnosis or yield prediction).

---

## 3. Architecture
 
```
                          STREAMLIT CHATBOT FRONTEND (streamlit_app.py)
                         [Image Upload | Chat Input | Suggested Queries]
                                            │
                                            ▼ HTTP
                                  FASTAPI BACKEND (src/main.py)
                                            │
                                            ▼
                               Input Agricultural Image
                                            │
                                            ▼
                                  RT-DETR-L Detector
                            (Ultralytics rtdetr-l.pt)
                                            │
                                            ▼
                                  Structured Detections
                        [class_id, class, bbox, confidence]
                                  ┌─────────┴─────────┐
                                  │                   │
                                  ▼                   ▼
                               /detect             /reason
                                                      │
                                                      ▼
                                              Intent Classifier
                                            (Deterministic Regex)
                                                      │
                                      ┌───────────────┴───────────────┐
                                      │                               │
                                  Supported                      Unsupported
                                      │                               │
                                      ▼                               ▼
                               Confidence Guard           "Insufficient information"
                                      │                     (Out of scope query)
                              ┌───────┴───────┐
                              │               │
                          Sufficient    Insufficient
                              │               │
                              ▼               ▼
                          Reasoning    "Insufficient information"
                           Engine      (Unconfirmed detections)
                              │
                              ▼
                       Structured Answer
```

---

## 4. Dataset & Provenance

### Dataset Overview
- **Source Dataset**: [Kaggle: Crop and Weed Detection Data with Bounding Boxes](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes)
- **Local Source Archive**: `archive.zip` located at workspace root.
- **Domain**: In-field agricultural imagery containing mixed crops and weeds under natural sunlight and soil conditions.
- **Classes**:
  - `0: crop` — Cultivated crop plants.
  - `1: weed` — Competing wild weed vegetation.
- **Annotations**: YOLO format (`<class_id> <x_center> <y_center> <width> <height>` normalized to `[0, 1]`).

### Verified Dataset Statistics
- **Total Images**: 1,300 JPEG images
- **Total Annotations**: 2,072 bounding boxes
- **Crop Annotations (Class 0)**: 1,212 (58.5%)
- **Weed Annotations (Class 1)**: 860 (41.5%)
- **Images with Crops**: 635 images
- **Images with Weeds**: 667 images
- **Images with Both Classes**: 2 images
- **Corrupt / Duplicate Images**: 0 (100% verified integrity)
- **Bounding Box Geometry**: Mean normalized width = 0.5093, Mean normalized height = 0.4611, Mean aspect ratio ($w/h$) = 1.1461.

---

## 5. Dataset Preparation & Split

Dataset preparation is handled deterministically by `src/prepare_dataset.py`:
1. Safely inspects and extracts `archive.zip`.
2. Validates image readability, format, dimensions, and MD5 file hashes.
3. Validates YOLO annotation boundaries, tokens, and class IDs.
4. Performs a fixed random split with **Seed = 42**:
   - **Train Set (70%)**: 910 images | 1,439 bounding boxes (848 crops, 591 weeds)
   - **Validation Set (15%)**: 195 images | 282 bounding boxes (157 crops, 125 weeds)
   - **Test Set (Held-Out, 15%)**: 195 images | 351 bounding boxes (207 crops, 144 weeds)
5. Generates `data/data.yaml` configured to resolve the dataset
   directories from the project root and generates
   `data/dataset_info.md`.

---

## 6. Model Architecture: RT-DETR-L

AgriVision utilizes **Ultralytics RT-DETR-L** (`rtdetr-l.pt`), a Real-Time DEtection TRansformer fine-tuned for agricultural object detection:
- **Backbone**: Efficient hybrid encoder (HGStem and HGBlock) extracting multi-scale feature maps.
- **Intra-scale Feature Interaction (AIFI)**: Single-scale Transformer encoder operating on high-level feature maps to reduce computational complexity.
- **Cross-scale Feature-fusion (CCFM)**: RepC3-based fusion blocks aggregating contextual semantic features.
- **Decoder**: RTDETRDecoder query-based head with Hungarian bipartite matching loss (GIoU loss + Classification focal loss + L1 bounding box loss).
- **Parameters**: 32.81 Million parameters.
- **Trained Model Checkpoint**: Saved at `models/best.pt`.

---

## 7. Training Configuration & Hardware

Training was executed via `src/train.py` with the following parameters and recorded in `outputs/training_metadata.json`:

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Model** | RT-DETR-L (`rtdetr-l.pt`) | Base transformer architecture |
| **Epochs** | 5 | Fine-tuning run on 2-class dataset |
| **Input Image Size (`imgsz`)** | 384 | Spatial resolution for CPU training |
| **Batch Size (`batch`)** | 8 | Batch size |
| **Patience** | 8 | Early stopping threshold |
| **Random Seed** | 42 | Deterministic seed |
| **Device** | CPU | Intel Core i7 (12 logical cores, 31.65 GB RAM) |
| **Training Duration** | ~18,692 seconds | ~5.19 hours CPU compute |
| **Artifact Checkpoint** | `models/best.pt` | Preserved best weights (~66.5 MB) |

---


## 8. Evaluation & Measured Results

Evaluation was performed on the held-out test split using the trained
RT-DETR-L checkpoint at `models/best.pt`. The test set contains
195 images and 351 annotated object instances.

### Measured Test Metrics

| Metric | Result |
| :--- | ---: |
| **Precision** | **71.65%** |
| **Recall** | **61.38%** |
| **mAP@50** | **68.09%** |
| **mAP@50-95** | **35.50%** |

### Per-Class Performance

| Class | AP@50 | AP@50-95 | Precision | Recall |
| :--- | ---: | ---: | ---: | ---: |
| **Crop** | **69.83%** | **35.70%** | **64.48%** | **68.60%** |
| **Weed** | **66.35%** | **35.29%** | **78.82%** | **54.17%** |

The metrics were obtained directly from the trained checkpoint using
`src/evaluate.py`. The machine-readable evaluation results are saved
in `outputs/evaluation/metrics.json`.

### Evaluation Methodology

The model was evaluated only on the held-out test split to avoid
reporting performance on training data. The evaluation reports
precision, recall, mAP@50, mAP@50-95, and per-class performance for
crop and weed.

The weed class has higher precision (78.82%) than recall (54.17%),
indicating that weed detections are relatively reliable when
produced, but the model still misses a meaningful portion of weed
instances.

*(Full machine-readable results are saved at `outputs/evaluation/metrics.json`.)*

---

## 9. Failure Case Analysis

Five representative failure cases were identified from the held-out
test set and are documented in `outputs/failure_cases/`.

### Case 1 — False Positive / Background Clutter

**Image:** `agri_0_1028.jpeg`

The test image contains 12 ground-truth crop objects, while the model
also produced an additional crop detection with confidence 0.55 that
did not correspond to a ground-truth object.

This represents a background-related false positive. Visually
complex field regions may contain patterns similar to learned crop
features.

**Potential improvement:** Hard-negative mining using soil,
non-crop vegetation, and background regions, together with
multi-scale augmentation and higher-resolution inference.

### Case 2 — False Negative / Missed Crop Detection

**Image:** `agri_0_1094.jpeg`

The image contains one ground-truth crop, but the detector failed to
produce a corresponding detection. The annotated object has an image
space size of approximately 424 × 398 pixels.

This indicates difficulty distinguishing the crop from its surrounding
scene despite the object having a substantial image-space size.

**Potential improvement:** Additional representative training
examples, targeted augmentation for lighting and occlusion, and
higher-resolution inference.

### Case 3 — False Positive / Background Clutter

**Image:** `agri_0_126.jpeg`

The image contains one ground-truth weed, while the detector produced
a spurious weed detection with confidence 0.75 that did not correspond
to a ground-truth object.

This indicates that visually similar vegetation or background
patterns can activate weed-related features.

**Potential improvement:** Hard-negative mining with non-weed
vegetation and background regions, together with confidence-threshold
tuning using validation data.

### Case 4 — Crop-versus-Weed Class Confusion

**Image:** `agri_0_1494.jpeg`

The image contains one ground-truth weed, but the model classified
the detected object as crop with high confidence (0.91).

This demonstrates the difficulty of separating visually similar
crop and weed vegetation.

**Potential improvement:** Add visually similar crop and weed
examples, targeted augmentation, and higher-resolution training or
inference.

### Case 5 — False Positive / Background Complexity

**Image:** `agri_0_1499.jpeg`

The image contains 11 ground-truth crop objects, while the detector
produced an additional crop detection with confidence 0.70 that did
not correspond to a ground-truth object.

This indicates that a background or vegetation region was sufficiently
similar to learned crop features to pass the detection threshold.

**Potential improvement:** Add hard-negative samples from complex
field backgrounds, apply multi-scale augmentation, and tune the
confidence threshold against validation-set precision and recall.

Detailed machine-readable failure analysis is available in
`outputs/failure_cases/failure_analysis.json`.

---

## 10. Reasoning Layer & Confidence Guardrails

The reasoning engine (`src/reasoning.py`) implements deterministic Python routing without external LLMs:

### Supported Intents
1. **COUNT**: `"How many weeds are present?"` $\rightarrow$ Returns exact count or guardrail.
2. **COMPARISON**: `"Are there more crops or weeds?"` $\rightarrow$ Compares counts if both classes are confirmed.
3. **PRESENCE**: `"Are there any weeds?"` $\rightarrow$ Confirms presence or reports insufficient evidence.
4. **SUMMARY**: `"What objects are visible?"` $\rightarrow$ Summarizes detected classes and totals.
5. **RATIO**: `"What is the crop-to-weed ratio?"` $\rightarrow$ Calculates simplified integer & float ratio, safely handling zero division.

### Strict Guardrail Rules
- **No Hallucinated Absence**: If 0 confident weed detections are found, the system does **not** assert "There are 0 weeds." Instead, it responds:
  > *"Insufficient information. The detector did not produce sufficiently confident weed detections."*
- **Comparison Guardrail**: If one class has zero confident detections, the system refuses to guarantee which is more numerous.
- **Unsupported Queries**: For out-of-scope agronomic advice (disease, fertilizer, yield, species, weather):
  > *"Insufficient information. The available detector only identifies crops and weeds and does not provide enough information to answer this question."*

---

## 11. REST API Specification

Built with **FastAPI** (`src/main.py`), providing clean JSON responses with standard Python logging.

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API service metadata |
| `GET` | `/health` | Model readiness and status |
| `POST` | `/detect` | Runs RT-DETR-L on image file, returns bounding boxes and counts |
| `POST` | `/reason` | Accepts image file + question, returns guarded reasoning answer |

### API Usage Examples

#### 1. Object Detection (`POST /detect`)
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "image=@data/images/test/agri_0_1017.jpeg"
```
**Response**:
```json
{
  "detections": [
    {
      "class_id": 1,
      "class": "weed",
      "confidence": 0.8732,
      "bbox": [120, 50, 220, 180]
    }
  ],
  "counts": {
    "crop": 0,
    "weed": 1
  },
  "total_count": 1,
  "inference_time_ms": 15.2
}
```

#### 2. Natural Language Reasoning (`POST /reason`)
```bash
curl -X POST "http://localhost:8000/reason" \
  -F "image=@data/images/test/agri_0_1017.jpeg" \
  -F "question=How many weeds are present?"
```
**Response**:
```json
{
  "question": "How many weeds are present?",
  "intent": "count",
  "answer": "I detected 1 weed.",
  "confidence": "high",
  "evidence": {
    "weed_count": 1,
    "confidence_threshold": 0.5
  }
}
```

#### 3. Unsupported Query Refusal (`POST /reason`)
```bash
curl -X POST "http://localhost:8000/reason" \
  -F "image=@data/images/test/agri_0_1017.jpeg" \
  -F "question=What fertilizer should I use for this field?"
```
**Response**:
```json
{
  "question": "What fertilizer should I use for this field?",
  "intent": "unsupported",
  "answer": "Insufficient information. The available detector only identifies crops and weeds and does not provide enough information to answer this question.",
  "confidence": "insufficient",
  "evidence": {
    "reason": "out_of_scope_intent",
    "supported_intents": ["count", "comparison", "presence", "summary", "ratio"]
  }
}
```

---

## 12. Installation & Reproduction Guide

### 1. Environment Setup
```bash
# Clone and enter workspace
cd agriculture_object_detection

# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Install exact dependencies
pip install -r requirements.txt
```

### 2. Prepare & Validate Dataset
```bash
# Extract archive, verify integrity, and create deterministic split
python src/prepare_dataset.py

# Run comprehensive dataset validation
python src/validate_dataset.py
```

### 3. Training

The submitted RT-DETR-L checkpoint was trained under limited CPU
compute using the following configuration:

```bash
python src/train.py \
  --data data/data.yaml \
  --epochs 5 \
  --imgsz 384 \
  --batch 8 \
  --device cpu \
  --seed 42 \
  --patience 8
```

### 4. Evaluation
```bash
# Evaluate on held-out test split
python src/evaluate.py --model models/best.pt --data data/data.yaml
```

### 5. Run Test Suite

```bash
pytest tests/ -v
```

The current test suite contains API, detector, and reasoning tests.

**Test result: 45 passed, 2 warnings.**

The warnings are dependency deprecation warnings and do not represent test failures.

### 6. Launch FastAPI Server
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 7. Launch Streamlit Chatbot Frontend
In a second terminal:
```bash
streamlit run streamlit_app.py
```
Open `http://localhost:8501` in your browser to interact with the visual chatbot interface.

---

## 13. Hardware & Runtime Specifications
- **Operating System**: Windows (AMD64)
- **Python Version**: 3.11.15
- **PyTorch Version**: 2.5.1
- **Ultralytics Version**: 8.4.148
- **Compute Device**: Intel Core i7 (CPU execution) / CUDA auto-detectable
- **Exact Metadata**: Recorded dynamically in `outputs/training_metadata.json`

---

## 14. Real-World Limitations
The current model achieves 68.09% mAP@50 and 35.50% mAP@50-95 on
the held-out test set. Weed recall (54.17%) is lower than weed
precision (78.82%), indicating that the detector is relatively
conservative when identifying weeds and still misses a portion of
weed instances.

Training was performed under limited CPU compute, which constrained
the practical training budget. Further training, higher-resolution
inference, additional representative samples, hard-negative mining,
and targeted augmentation could potentially improve detection,
particularly weed recall and difficult background cases.
1. **Domain Shift & Lighting**: In-field lighting varies dramatically with cloud cover, golden hour sun, and wet soil reflection.
2. **Binary Class Granularity**: The detector classifies vegetation broadly into `crop` and `weed` without fine-grained botanical species taxonomy.
3. **Non-Calibrated Probabilities**: Softmax detector confidence scores reflect bounding box ranking rather than calibrated Bayesian posterior probabilities.
4. **Resolution Constraints**: Downsampling high-resolution field drone/tractor camera feeds to 640px may obscure newly germinated weed seedlings.
