# AgriVision: Technical Memorandum

**To**: Technical Screening Review Committee  
**From**: Agricultural Vision Engineering Team  
**Subject**: Architecture, Evaluation, and Failure Analysis of the AgriVision RT-DETR Crop & Weed Detection and Reasoning System  
**Date**: September 13, 2026  
**Status**: Technical Screening Prototype  

---

## 1. Problem & Objective
In precision agriculture, automated weed management and crop monitoring require reliable spatial localization of cultivated crops and competing weed vegetation. Differentiating between crops and weeds enables site-specific intervention, such as precision spraying and mechanical cultivation, reducing chemical usage and input costs. 

Key visual challenges in agricultural environments include morphological similarities between crop plants and weed seedlings, canopy occlusion, variable illumination, and complex soil backgrounds. AgriVision addresses these challenges by integrating an **Ultralytics RT-DETR-L** real-time transformer detector with a **deterministic, confidence-guarded reasoning layer** exposed via high-performance FastAPI and Streamlit interfaces.

---

## 2. Dataset & Preparation
- **Source**: [Kaggle: Crop and Weed Detection Data with Bounding Boxes](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes), extracted from `archive.zip`.
- **Dataset Properties**: 1,300 in-field JPEG images with 2,072 normalized bounding boxes in YOLO format (`<class_id> <x_center> <y_center> <width> <height>`).
  - `Class 0 (Crop)`: 1,212 instances (58.5%).
  - `Class 1 (Weed)`: 860 instances (41.5%).
  - **Quality Audit**: Verified 0 corrupt images, 0 duplicate MD5 hashes, 0 out-of-bounds coordinates, and 100% label validity.
- **Deterministic 70/15/15 Split (`seed = 42`)**:
  - **Train (70%)**: 910 images | 1,439 bounding boxes (848 crops, 591 weeds)
  - **Validation (15%)**: 195 images | 282 bounding boxes (157 crops, 125 weeds)
  - **Held-Out Test (15%)**: 195 images | 351 bounding boxes (207 crops, 144 weeds)
  - The held-out test split remained strictly isolated during development and was utilized exclusively for final metric evaluation and failure analysis.

---

## 3. RT-DETR-L Model Architecture
AgriVision utilizes **Ultralytics RT-DETR-L** (`rtdetr-l.pt`), a Real-Time DEtection TRansformer with approximately 32 Million parameters (31,987,850 parameters):
- **Backbone**: Efficient hybrid encoder (HGStem and HGBlock) extracting multi-scale feature representations.
- **Intra-scale Feature Interaction (AIFI)**: Single-scale Transformer encoder operating on high-level feature maps to reduce computational overhead.
- **Cross-scale Feature-fusion (CCFM)**: RepC3-based fusion blocks aggregating contextual semantic features across spatial resolutions.
- **Decoder**: RTDETRDecoder query-based head with Hungarian bipartite matching loss (GIoU loss + Focal Classification loss + L1 coordinate loss), eliminating non-maximum suppression (NMS) latency bottlenecks.

---

## 4. Training Configuration
Model fine-tuning was executed via `src/train.py` with full reproducibility metadata saved in `outputs/training_metadata.json`:
- **Model**: RT-DETR-L (`rtdetr-l.pt`)
- **Epochs**: 5
- **Input Image Size (`imgsz`)**: 384
- **Batch Size (`batch`)**: 8
- **Device**: CPU
- **Random Seed**: 42
- **Patience**: 8
- **Training Duration**: ~18,692 seconds (~5.19 hours)
- **Saved Final Checkpoint**: `models/best.pt` (~66.5 MB)

---

## 5. Hardware & Runtime Constraints
Training was conducted entirely on CPU (Intel Core i7 with 12 logical cores and 31.65 GB RAM) under limited computational budget. This compute constraint governed the choice of 5 fine-tuning epochs and an input resolution of 384 pixels.

---

## 6. Evaluation Methodology
Evaluation was performed strictly on the **held-out test split** (195 images, 351 annotated object instances) using `src/evaluate.py` against `models/best.pt`. No test split data was seen during training or threshold selection. Standard COCO evaluation protocols were used to compute Precision, Recall, mAP@50, and mAP@50-95.

---

## 7. Actual Evaluation Results
The test metrics recorded in `outputs/evaluation/metrics.json` are:

| Metric | Result |
| :--- | ---: |
| **Precision** | **71.65%** |
| **Recall** | **61.38%** |
| **mAP@50** | **68.09%** |
| **mAP@50-95** | **35.50%** |

---

## 8. Per-Class Performance

| Class | Instances | AP@50 | AP@50-95 | Precision | Recall |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **Crop (Class 0)** | 207 | **69.83%** | **35.70%** | **64.48%** | **68.60%** |
| **Weed (Class 1)** | 144 | **66.35%** | **35.29%** | **78.82%** | **54.17%** |

### Performance Analysis:
- **Overall Balance**: The detector achieves 68.09% mAP@50 and 35.50% mAP@50-95 on the held-out test split. Overall precision (71.65%) exceeds overall recall (61.38%).
- **Weed Class Behavior**: Weed detections exhibit high precision (**78.82%**) but moderate recall (**54.17%**). When the model predicts a weed, the detection is relatively reliable; however, the detector misses approximately 45.8% of ground-truth weed instances.
- **Crop Class Behavior**: Crop detection shows more balanced precision (**64.48%**) and recall (**68.60%**), reflecting the larger number of crop training instances.

---

## 9. Failure-Case Analysis
Five representative failure cases from the held-out test set are documented in `outputs/failure_cases/failure_analysis.json`:

1. **Case 1 (`agri_0_1028.jpeg`) — Background-Related False Positive**:
   - *Observation*: Image contains 12 ground-truth crops; model produced an additional spurious crop detection (confidence 0.55).
   - *Analysis*: Complex soil texture and organic debris activated learned crop feature patterns above the threshold.
2. **Case 2 (`agri_0_1094.jpeg`) — Missed Crop Detection (False Negative)**:
   - *Observation*: Ground-truth crop object (size 424 × 398 pixels) was missed by the detector.
   - *Analysis*: Despite large image-space size, subtle visual contrast against surrounding soil and diffuse lighting hindered detection.
3. **Case 3 (`agri_0_126.jpeg`) — Background-Related False Positive**:
   - *Observation*: Image contains 1 ground-truth weed; model produced a spurious weed detection (confidence 0.75).
   - *Analysis*: Visually textured background patterns resembled weed foliage features.
4. **Case 4 (`agri_0_1494.jpeg`) — Crop-versus-Weed Class Confusion**:
   - *Observation*: Ground-truth weed was predicted as crop with confidence 0.91.
   - *Analysis*: Morphological leaf similarity and canopy overlap caused the model to misassign the plant class.
5. **Case 5 (`agri_0_1499.jpeg`) — Background-Related False Positive**:
   - *Observation*: Image contains 11 ground-truth crops; model generated a spurious crop detection (confidence 0.70).
   - *Analysis*: Soil clutter and background foliage patterns were sufficiently similar to crop features to pass the operating threshold.

---

## 10. Reasoning Layer & Confidence Guardrails
To eliminate hallucinations common in generative LLMs, AgriVision implements a **deterministic, rule-based reasoning engine** (`src/reasoning.py`):
- **Intent Routing**: Classifies queries into supported intents (`COUNT`, `COMPARISON`, `PRESENCE`, `SUMMARY`, `RATIO`) and catches unsupported queries (`UNSUPPORTED`).
- **Evidence Binding**: Reasoning operates strictly on confidence-filtered detections ($conf \ge 0.50$).
- **Absence of Evidence Guardrail**: When 0 confident detections are present for a class, the system refuses to assert absolute absence, returning:
  > *"Insufficient information. The detector did not produce sufficiently confident weed detections."*
- **Scope Guardrail**: Out-of-scope agronomic queries (e.g., fertilizer dosage, disease diagnosis, yield estimation) return:
  > *"Insufficient information. The available detector only identifies crops and weeds and does not provide enough information to answer this question."*

---

## 11. API & System Design
- **FastAPI Backend (`src/main.py`)**: Exposes structured JSON endpoints:
  - `GET /health`: Service health and model status.
  - `POST /detect`: Returns bounding boxes, confidence scores, and class counts.
  - `POST /reason`: Processes image + natural language query and returns guarded structured reasoning.
- **Streamlit Frontend (`streamlit_app.py`)**: A chat-style interface communicating strictly over HTTP with FastAPI, visualizing detections and evidence.

---

## 12. Real-World Limitations
1. **Weed Recall Gap**: Weed recall (54.17%) leaves a portion of weeds undetected, which could allow competing weeds to remain unmanaged in autonomous field operations.
2. **Binary Taxonomic Granularity**: Broad classification into `crop` and `weed` does not identify specific weed species or crop varieties required for specialized herbicide selection.
3. **Domain & Environmental Shift**: Field performance may vary under extreme lighting changes, wet soil reflections, or different crop growth stages not present in the training distribution.
4. **Resolution Constraints**: 384px input resolution may obscure newly germinated, small weed seedlings.

---

## 13. Proposed Future Improvements
The following improvements are proposed for future development phases (not applied to the current prototype):
1. **Hard-Negative Mining**: Training on explicit background and soil-clutter patches to suppress false positive detections.
2. **Targeted Data Augmentation**: Multi-scale cropping, illumination jitter, and foliage occlusion synthesis to improve small-plant localization and contrast robustness.
3. **Representative Difficult Examples**: Expanding dataset coverage with ambiguous weed species and early-stage emergence imagery.
4. **Confidence-Threshold Tuning**: Optimizing per-class operating thresholds on validation data to balance precision and recall.
5. **Higher-Resolution GPU Training**: Fine-tuning at 640px or higher spatial resolution on GPU hardware to improve small-object feature resolution.
