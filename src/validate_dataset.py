"""
Dataset validation module for AgriVision.

Validates images, labels, pairings, and geometric boundaries across train, val, and test splits.
Computes bounding-box dimension and aspect ratio statistics.
Exits with non-zero code if any critical integrity issues are detected.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_dataset")


def validate_split(data_dir: Path, split: str) -> Tuple[Dict, List[str]]:
    """
    Validates images and labels within a specific split (train, val, test).
    """
    img_dir = data_dir / "images" / split
    lbl_dir = data_dir / "labels" / split
    errors = []

    if not img_dir.exists():
        errors.append(f"Image directory not found: {img_dir}")
        return {}, errors
    if not lbl_dir.exists():
        errors.append(f"Label directory not found: {lbl_dir}")
        return {}, errors

    img_files = sorted([f for f in img_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    lbl_files = sorted([f for f in lbl_dir.glob("*.txt")])

    img_stems = {f.stem: f for f in img_files}
    lbl_stems = {f.stem: f for f in lbl_files}

    # Pairing checks
    missing_labels = [f for stem, f in img_stems.items() if stem not in lbl_stems]
    orphan_labels = [f for stem, f in lbl_stems.items() if stem not in img_stems]

    if missing_labels:
        for m in missing_labels:
            errors.append(f"[{split}] Image missing label file: {m.name}")
    if orphan_labels:
        for o in orphan_labels:
            errors.append(f"[{split}] Orphan label file without image: {o.name}")

    total_images = len(img_files)
    total_boxes = 0
    crop_boxes = 0
    weed_boxes = 0
    images_with_crop = 0
    images_with_weed = 0
    images_with_both = 0
    widths = []
    heights = []
    aspect_ratios = []

    for stem, img_path in img_stems.items():
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            errors.append(f"[{split}] Unreadable/corrupt image: {img_path.name}")
            continue

        h, w, c = img.shape
        if h <= 0 or w <= 0 or c != 3:
            errors.append(f"[{split}] Invalid image shape {img.shape}: {img_path.name}")
            continue

        lbl_path = lbl_stems.get(stem)
        if not lbl_path:
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            errors.append(f"[{split}] Empty label file: {lbl_path.name}")
            continue

        has_crop = False
        has_weed = False

        for line_idx, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"[{split}] Line {line_idx} in {lbl_path.name} has {len(parts)} tokens (expected 5)")
                continue

            try:
                cid = int(parts[0])
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError as e:
                errors.append(f"[{split}] Line {line_idx} in {lbl_path.name} non-numeric: {e}")
                continue

            if cid not in (0, 1):
                errors.append(f"[{split}] Line {line_idx} in {lbl_path.name} invalid class ID: {cid}")

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                errors.append(f"[{split}] Line {line_idx} in {lbl_path.name} center out of range: ({xc}, {yc})")

            if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                errors.append(f"[{split}] Line {line_idx} in {lbl_path.name} dimensions out of range: ({bw}, {bh})")

            total_boxes += 1
            if cid == 0:
                crop_boxes += 1
                has_crop = True
            elif cid == 1:
                weed_boxes += 1
                has_weed = True

            widths.append(bw)
            heights.append(bh)
            if bh > 0:
                aspect_ratios.append(bw / bh)

        if has_crop and has_weed:
            images_with_both += 1
        if has_crop:
            images_with_crop += 1
        if has_weed:
            images_with_weed += 1

    stats = {
        "split": split,
        "images": total_images,
        "total_boxes": total_boxes,
        "crop_boxes": crop_boxes,
        "weed_boxes": weed_boxes,
        "images_with_crop": images_with_crop,
        "images_with_weed": images_with_weed,
        "images_with_both": images_with_both,
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios,
    }
    return stats, errors


def validate_dataset(data_dir: str = "data") -> bool:
    """
    Runs comprehensive validation across train, val, and test splits.
    """
    base_dir = Path(data_dir).resolve()
    logger.info(f"Starting dataset validation for directory: {base_dir}")

    yaml_file = base_dir / "data.yaml"
    if not yaml_file.exists():
        logger.error(f"Missing data.yaml at {yaml_file}")
        return False

    all_errors: List[str] = []
    split_stats = {}

    for split in ["train", "val", "test"]:
        stats, errs = validate_split(base_dir, split)
        split_stats[split] = stats
        all_errors.extend(errs)

    # Print Report
    print("\n" + "=" * 60)
    print("AGRIVISION DATASET VALIDATION REPORT")
    print("=" * 60)

    total_imgs = sum(s.get("images", 0) for s in split_stats.values())
    total_boxes = sum(s.get("total_boxes", 0) for s in split_stats.values())
    total_crops = sum(s.get("crop_boxes", 0) for s in split_stats.values())
    total_weeds = sum(s.get("weed_boxes", 0) for s in split_stats.values())

    print(f"Total Dataset Images:     {total_imgs}")
    print(f"  - Train split:          {split_stats['train'].get('images', 0)} ({split_stats['train'].get('images', 0)/total_imgs*100:.1f}%)")
    print(f"  - Val split:            {split_stats['val'].get('images', 0)} ({split_stats['val'].get('images', 0)/total_imgs*100:.1f}%)")
    print(f"  - Test split:           {split_stats['test'].get('images', 0)} ({split_stats['test'].get('images', 0)/total_imgs*100:.1f}%)")
    print("-" * 60)
    print(f"Total Bounding Boxes:     {total_boxes}")
    print(f"  - Crop Boxes (Class 0): {total_crops} ({total_crops/total_boxes*100:.1f}%)")
    print(f"  - Weed Boxes (Class 1): {total_weeds} ({total_weeds/total_boxes*100:.1f}%)")
    print(f"  - Avg Boxes/Image:      {total_boxes/total_imgs:.2f}")
    print("-" * 60)

    all_widths = []
    all_heights = []
    all_ar = []
    for s in split_stats.values():
        all_widths.extend(s.get("widths", []))
        all_heights.extend(s.get("heights", []))
        all_ar.extend(s.get("aspect_ratios", []))

    if all_widths:
        w_arr = np.array(all_widths)
        h_arr = np.array(all_heights)
        ar_arr = np.array(all_ar)
        print("Bounding Box Geometry (Normalized Dimensions):")
        print(f"  - Width:  min={w_arr.min():.4f}, mean={w_arr.mean():.4f}, max={w_arr.max():.4f}")
        print(f"  - Height: min={h_arr.min():.4f}, mean={h_arr.mean():.4f}, max={h_arr.max():.4f}")
        print(f"  - Aspect Ratio (w/h): min={ar_arr.min():.4f}, mean={ar_arr.mean():.4f}, max={ar_arr.max():.4f}")
    print("-" * 60)

    for split in ["train", "val", "test"]:
        s = split_stats[split]
        print(f"[{split.upper()}] Images: {s.get('images', 0)} | Boxes: {s.get('total_boxes', 0)} "
              f"(Crops: {s.get('crop_boxes', 0)}, Weeds: {s.get('weed_boxes', 0)}) | "
              f"Crop Imgs: {s.get('images_with_crop', 0)}, Weed Imgs: {s.get('images_with_weed', 0)}, Both: {s.get('images_with_both', 0)}")

    print("=" * 60)

    if all_errors:
        print(f"\nCRITICAL ERRORS DETECTED ({len(all_errors)}):")
        for err in all_errors[:20]:
            print(f"  [!] {err}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more errors.")
        print("=" * 60 + "\n")
        return False

    print("\nSTATUS: ALL VALIDATION CHECKS PASSED SUCCESSFULLY (0 Errors)\n")
    print("=" * 60 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate processed YOLO agricultural dataset.")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory containing images and labels")
    args = parser.parse_args()

    success = validate_dataset(data_dir=args.data_dir)
    if not success:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
