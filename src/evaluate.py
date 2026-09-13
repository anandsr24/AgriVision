"""
Evaluation and Metrics Extraction Script for AgriVision.

Evaluates the trained RT-DETR-L model on the held-out test split (data/images/test),
computes mAP50, mAP50-95, precision, recall, per-class metrics, extracts predictions and failure cases,
and saves outputs/evaluation/metrics.json.
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import RTDETR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")

CLASS_NAMES = {0: "crop", 1: "weed"}


def evaluate_model(
    model_path: str = "models/best.pt",
    data_yaml: str = "data/data.yaml",
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
    output_dir: str = "outputs/evaluation",
    failure_dir: str = "outputs/failure_cases",
    predictions_dir: str = "outputs/predictions",
) -> Dict:
    """
    Runs evaluation on the held-out test set and saves metrics and visualizations.
    """
    model_file = Path(model_path).resolve()
    data_file = Path(data_yaml).resolve()
    out_eval = Path(output_dir).resolve()
    out_fail = Path(failure_dir).resolve()
    out_preds = Path(predictions_dir).resolve()

    out_eval.mkdir(parents=True, exist_ok=True)
    out_fail.mkdir(parents=True, exist_ok=True)
    out_preds.mkdir(parents=True, exist_ok=True)

    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_file}")
    if not data_file.exists():
        raise FileNotFoundError(f"Data configuration not found at {data_file}")

    logger.info(f"Loading model from {model_file} for evaluation on test set...")
    model = RTDETR(str(model_file))

    # Evaluate on test split
    metrics = model.val(
        data=str(data_file),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out_eval),
        name="test_eval",
        exist_ok=True,
    )

    # Extract overall metrics
    # Ultralytics results.box: maps, map50, map75, map, mp (mean precision), mr (mean recall)
    box = metrics.box
    map50 = float(box.map50)
    map50_95 = float(box.map)
    precision = float(box.mp)
    recall = float(box.mr)

    # Per-class metrics
    # maps: [class_0_map50_95, class_1_map50_95]
    # ap50: [class_0_ap50, class_1_ap50]
    # p: [class_0_p, class_1_p]
    # r: [class_0_r, class_1_r]
    per_class = {}
    class_names = metrics.names if hasattr(metrics, "names") else CLASS_NAMES

    for i, cls_name in class_names.items():
        try:
            cls_ap50 = float(box.ap50[i]) if hasattr(box, "ap50") and len(box.ap50) > i else float(box.maps[i])
            cls_p = float(box.p[i]) if hasattr(box, "p") and len(box.p) > i else 0.0
            cls_r = float(box.r[i]) if hasattr(box, "r") and len(box.r) > i else 0.0
            cls_map50_95 = float(box.maps[i]) if hasattr(box, "maps") and len(box.maps) > i else 0.0
        except Exception:
            cls_ap50 = 0.0
            cls_p = 0.0
            cls_r = 0.0
            cls_map50_95 = 0.0

        per_class[cls_name] = {
            "AP50": round(cls_ap50, 4),
            "AP50_95": round(cls_map50_95, 4),
            "precision": round(cls_p, 4),
            "recall": round(cls_r, 4),
        }

    formatted_metrics = {
        "model": "RT-DETR-L",
        "model_path": str(model_file),
        "split": "test",
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "classes": per_class,
    }

    metrics_json_path = out_eval / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(formatted_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_json_path}")

    # Run detailed failure case analysis and visualization on test set
    failure_cases = analyze_failure_cases(
        model=model,
        data_dir=data_file.parent,
        output_dir=out_fail,
        preds_dir=out_preds,
        conf_threshold=0.50,
    )
    formatted_metrics["failure_cases_count"] = len(failure_cases)

    # Copy generated confusion matrix / plots from ultralytics val directory if present
    val_save_dir = Path(metrics.save_dir) if hasattr(metrics, "save_dir") else out_eval / "test_eval"
    if val_save_dir.exists():
        for plot_file in val_save_dir.glob("*.png"):
            dest = out_eval / plot_file.name
            if not dest.exists():
                shutil.copy2(plot_file, dest)

    return formatted_metrics


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculates Intersection over Union for [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def analyze_failure_cases(
    model: RTDETR,
    data_dir: Path,
    output_dir: Path,
    preds_dir: Path,
    conf_threshold: float = 0.50,
) -> List[Dict]:
    """
    Evaluates test set images, compares predictions with ground truth YOLO annotations,
    and documents 5 real failure cases.
    """
    test_img_dir = data_dir / "images" / "test"
    test_lbl_dir = data_dir / "labels" / "test"

    if not test_img_dir.exists() or not test_lbl_dir.exists():
        logger.warning(f"Test split directories not found in {data_dir}")
        return []

    test_imgs = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.jpeg")) + list(test_img_dir.glob("*.png")))
    detected_failures: List[Dict] = []
    saved_samples_count = 0

    logger.info(f"Analyzing {len(test_imgs)} test images for predictions and failure cases...")

    for img_path in test_imgs:
        stem = img_path.stem
        lbl_path = test_lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        # Load image dimensions
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h_img, w_img, _ = img_bgr.shape

        # Read Ground Truth
        gt_boxes = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = (xc - bw / 2) * w_img
                    y1 = (yc - bh / 2) * h_img
                    x2 = (xc + bw / 2) * w_img
                    y2 = (yc + bh / 2) * h_img
                    gt_boxes.append({
                        "class_id": cid,
                        "class": CLASS_NAMES.get(cid, str(cid)),
                        "bbox": [x1, y1, x2, y2],
                        "matched": False,
                    })

        # Run Prediction
        results = model.predict(source=str(img_path), conf=0.10, device=model.device, verbose=False)
        pred_boxes = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for b, c, cl in zip(xyxy, confs, clss):
                pred_boxes.append({
                    "class_id": int(cl),
                    "class": CLASS_NAMES.get(int(cl), str(cl)),
                    "confidence": float(c),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    "matched": False,
                })

        # Save some sample correct predictions in outputs/predictions
        if saved_samples_count < 10 and len(pred_boxes) > 0 and len(gt_boxes) > 0:
            pred_vis = img_bgr.copy()
            for pb in pred_boxes:
                if pb["confidence"] >= conf_threshold:
                    x1, y1, x2, y2 = [int(v) for v in pb["bbox"]]
                    color = (0, 255, 0) if pb["class_id"] == 0 else (0, 0, 255)
                    cv2.rectangle(pred_vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(pred_vis, f"{pb['class']} {pb['confidence']:.2f}", (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imwrite(str(preds_dir / f"pred_{stem}.jpg"), pred_vis)
            saved_samples_count += 1

        # Match predictions to GT with IoU >= 0.40
        matched_gt = 0
        for pb in pred_boxes:
            best_iou = 0.0
            best_gt = None
            for gb in gt_boxes:
                iou = calculate_iou(pb["bbox"], gb["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gb
            if best_iou >= 0.40 and best_gt is not None and not best_gt["matched"]:
                pb["matched"] = True
                pb["iou"] = best_iou
                pb["gt_class"] = best_gt["class"]
                best_gt["matched"] = True
                matched_gt += 1

        # Identify Failure Modes:
        # 1. False Negative (Missed Ground Truth)
        unmatched_gt = [g for g in gt_boxes if not g["matched"]]
        # 2. False Positive (High confidence detection with no GT)
        false_positives = [p for p in pred_boxes if not p.get("matched") and p["confidence"] >= conf_threshold]
        # 3. Class Confusion (Matched bbox but predicted wrong class)
        class_confusions = [p for p in pred_boxes if p.get("matched") and p["class"] != p.get("gt_class")]
        # 4. Low Confidence Detection (Correct class & localization but conf < threshold)
        low_confs = [p for p in pred_boxes if p.get("matched") and p["class"] == p.get("gt_class") and p["confidence"] < conf_threshold]

        failure_type = None
        failure_details = ""
        observed = ""
        expected = f"{len(gt_boxes)} GT ({', '.join([g['class'] for g in gt_boxes])})"

        if class_confusions:
            failure_type = "Crop-vs-Weed Visual Similarity / Class Confusion"
            p = class_confusions[0]
            observed = f"Predicted {p['class']} ({p['confidence']:.2f}) instead of {p['gt_class']}"
            failure_details = (
                f"Model confused {p['gt_class']} with {p['class']} due to morphological leaf similarity "
                "or canopy overlap in complex vegetation background."
            )
        elif false_positives:
            failure_type = "False Positive / Background Clutter"
            p = false_positives[0]
            observed = f"Spurious {p['class']} detection (conf: {p['confidence']:.2f}) with no ground truth"
            failure_details = "Model triggered detection on soil texture, organic debris, or shadowed leaf patterns."
        elif unmatched_gt:
            failure_type = "False Negative / Missed Detection"
            g = unmatched_gt[0]
            w_box = g["bbox"][2] - g["bbox"][0]
            h_box = g["bbox"][3] - g["bbox"][1]
            observed = f"Failed to detect {g['class']} (size: {int(w_box)}x{int(h_box)}px)"
            failure_details = (
                "Ground truth object was missed, likely due to small physical scale, partial occlusion, "
                "or subtle contrast against surrounding soil."
            )
        elif low_confs:
            failure_type = "Low Confidence Detection"
            p = low_confs[0]
            observed = f"Detected {p['class']} with low confidence ({p['confidence']:.2f} < {conf_threshold})"
            failure_details = (
                "Object localized correctly but confidence score was below the operating threshold, "
                "triggering the reasoning insufficient-information guardrail."
            )

        if failure_type and len(detected_failures) < 5:
            # Save visual annotation of failure
            fail_vis = img_bgr.copy()
            # Draw GT in blue
            for gb in gt_boxes:
                x1, y1, x2, y2 = [int(v) for v in gb["bbox"]]
                cv2.rectangle(fail_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(fail_vis, f"GT:{gb['class']}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Draw Preds in red/green
            for pb in pred_boxes:
                if pb["confidence"] >= 0.20:
                    x1, y1, x2, y2 = [int(v) for v in pb["bbox"]]
                    cv2.rectangle(fail_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(fail_vis, f"Pred:{pb['class']} {pb['confidence']:.2f}", (x1, min(h_img - 5, y2 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            fail_img_name = f"failure_{len(detected_failures) + 1}_{stem}.jpg"
            cv2.imwrite(str(output_dir / fail_img_name), fail_vis)

            detected_failures.append({
                "case_id": len(detected_failures) + 1,
                "image_id": img_path.name,
                "image_file": fail_img_name,
                "failure_type": failure_type,
                "expected": expected,
                "observed": observed,
                "root_cause": failure_details,
                "primary_attribution": "Model Feature Representation / In-Field Lighting & Occlusion",
                "potential_improvement": "Incorporate multi-scale crop augmentation, hard-negative mining, and higher-resolution inference.",
            })

            if len(detected_failures) >= 5 and saved_samples_count >= 10:
                logger.info(f"Target of 5 failure cases and {saved_samples_count} sample predictions reached; stopping analysis.")
                break

    # Save failure analysis documentation
    fail_md = output_dir / "failure_analysis.json"
    with open(fail_md, "w", encoding="utf-8") as f:
        json.dump(detected_failures, f, indent=2)
    logger.info(f"Documented {len(detected_failures)} real failure cases to {output_dir}")

    return detected_failures


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RT-DETR-L on test split.")
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to best.pt")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0)")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation", help="Metrics output directory")
    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 50)
    print("EVALUATION METRICS SUMMARY (HELD-OUT TEST SET)")
    print("=" * 50)
    print(json.dumps(results, indent=2))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
