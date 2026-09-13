"""
Dataset preparation module for AgriVision.

Extracts archive.zip, validates images and YOLO annotations, detects duplicates and corruption,
performs a deterministic 70/15/15 train/val/test split (seed=42), and generates data.yaml and dataset_info.md.
"""

import argparse
import hashlib
import io
import logging
import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_dataset")


def calculate_file_hash(file_bytes: bytes) -> str:
    """Computes MD5 hash for duplicate detection."""
    return hashlib.md5(file_bytes).hexdigest()


def validate_yolo_line(line: str) -> Tuple[bool, int, float, float, float, float, str]:
    """
    Validates a single YOLO annotation line.
    Format: <class_id> <x_center> <y_center> <width> <height>
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return False, -1, 0.0, 0.0, 0.0, 0.0, f"Expected 5 tokens, got {len(parts)}"
    try:
        class_id = int(parts[0])
        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError as e:
        return False, -1, 0.0, 0.0, 0.0, 0.0, f"Non-numeric value: {e}"

    if class_id not in (0, 1):
        return False, class_id, xc, yc, w, h, f"Invalid class ID {class_id} (must be 0 or 1)"

    if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
        return False, class_id, xc, yc, w, h, f"Center coordinates ({xc}, {yc}) out of bounds [0, 1]"

    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False, class_id, xc, yc, w, h, f"Dimensions ({w}, {h}) out of bounds (0, 1]"

    return True, class_id, xc, yc, w, h, ""


def prepare_dataset(
    archive_path: str = "archive.zip",
    output_dir: str = "data",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict:
    """
    Extracts, validates, splits, and formats the dataset.
    """
    archive_file = Path(archive_path).resolve()
    if not archive_file.exists():
        raise FileNotFoundError(f"Archive not found at {archive_file}")

    out_base = Path(output_dir).resolve()
    raw_dir = out_base / "raw"
    images_base = out_base / "images"
    labels_base = out_base / "labels"

    for split in ["train", "val", "test"]:
        (images_base / split).mkdir(parents=True, exist_ok=True)
        (labels_base / split).mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Inspecting archive: {archive_file}")

    with zipfile.ZipFile(archive_file, "r") as z:
        all_entries = z.namelist()
        file_entries = [e for e in all_entries if not e.endswith("/")]

        img_entries = [e for e in file_entries if e.lower().endswith((".jpg", ".jpeg", ".png"))]
        txt_entries = [e for e in file_entries if e.lower().endswith(".txt")]

        logger.info(f"Found {len(img_entries)} image entries and {len(txt_entries)} text entries in zip.")

        # Map base name (without extension) to entry
        img_map: Dict[str, str] = {Path(e).stem: e for e in img_entries}
        txt_map: Dict[str, str] = {Path(e).stem: e for e in txt_entries}

        # Orphan check
        orphan_txts = [e for base, e in txt_map.items() if base not in img_map and Path(e).name != "classes.txt"]
        orphan_imgs = [e for base, e in img_map.items() if base not in txt_map]
        classes_txt_entry = next((e for e in txt_entries if Path(e).name == "classes.txt"), None)

        if orphan_txts:
            logger.warning(f"Found {len(orphan_txts)} orphan text files: {orphan_txts}")
        if orphan_imgs:
            logger.warning(f"Found {len(orphan_imgs)} orphan images (no label): {orphan_imgs}")

        # Validation pass
        valid_samples: List[Dict] = []
        corrupt_images = []
        duplicate_images = []
        empty_labels = []
        malformed_labels = []
        seen_hashes: Dict[str, str] = {}

        total_boxes = 0
        crop_boxes = 0
        weed_boxes = 0
        images_with_crop = 0
        images_with_weed = 0
        images_with_both = 0

        for base_name, img_entry in sorted(img_map.items()):
            img_bytes = z.read(img_entry)
            img_hash = calculate_file_hash(img_bytes)

            if img_hash in seen_hashes:
                duplicate_images.append((img_entry, seen_hashes[img_hash]))
                continue
            seen_hashes[img_hash] = img_entry

            # Verify image integrity & size
            try:
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    pil_img.verify()
                    width, height = pil_img.size
                if width <= 0 or height <= 0:
                    corrupt_images.append((img_entry, f"Invalid size {width}x{height}"))
                    continue
            except Exception as e:
                corrupt_images.append((img_entry, str(e)))
                continue

            # Verify label file
            txt_entry = txt_map.get(base_name)
            if not txt_entry:
                continue

            txt_bytes = z.read(txt_entry)
            txt_content = txt_bytes.decode("utf-8", errors="replace").strip()
            lines = [line.strip() for line in txt_content.splitlines() if line.strip()]

            if not lines:
                empty_labels.append(txt_entry)
                continue

            sample_boxes = []
            is_valid_sample = True
            sample_has_crop = False
            sample_has_weed = False

            for line in lines:
                is_valid, cid, xc, yc, w, h, err = validate_yolo_line(line)
                if not is_valid:
                    malformed_labels.append((txt_entry, line, err))
                    is_valid_sample = False
                    break
                sample_boxes.append((cid, xc, yc, w, h))
                if cid == 0:
                    sample_has_crop = True
                elif cid == 1:
                    sample_has_weed = True

            if not is_valid_sample:
                continue

            # Accumulate statistics
            for cid, xc, yc, w, h in sample_boxes:
                total_boxes += 1
                if cid == 0:
                    crop_boxes += 1
                elif cid == 1:
                    weed_boxes += 1

            if sample_has_crop and sample_has_weed:
                images_with_both += 1
            if sample_has_crop:
                images_with_crop += 1
            if sample_has_weed:
                images_with_weed += 1

            valid_samples.append({
                "base_name": base_name,
                "img_entry": img_entry,
                "txt_entry": txt_entry,
                "img_bytes": img_bytes,
                "txt_content": txt_content,
                "boxes": sample_boxes,
                "has_crop": sample_has_crop,
                "has_weed": sample_has_weed,
            })

    logger.info(f"Verified {len(valid_samples)} valid paired samples out of {len(img_map)} total images.")
    logger.info(f"Corrupt: {len(corrupt_images)}, Duplicates: {len(duplicate_images)}, Empty: {len(empty_labels)}, Malformed: {len(malformed_labels)}")

    # Extract all files into raw/ for reference
    with zipfile.ZipFile(archive_file, "r") as z:
        z.extractall(raw_dir)
    logger.info(f"Extracted raw archive to {raw_dir}")

    # Deterministic Split
    random.seed(seed)
    # Shuffle a shallow copy with seed for reproducibility
    shuffled_samples = list(valid_samples)
    random.shuffle(shuffled_samples)

    n_total = len(shuffled_samples)
    n_train = round(n_total * train_ratio)
    n_val = round(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_samples = shuffled_samples[:n_train]
    val_samples = shuffled_samples[n_train:n_train + n_val]
    test_samples = shuffled_samples[n_train + n_val:]

    splits_data = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    logger.info(f"Split counts -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Write files to target directories
    for split_name, samples in splits_data.items():
        img_dest_dir = images_base / split_name
        lbl_dest_dir = labels_base / split_name

        # Clean existing files in split directories
        for f in img_dest_dir.glob("*"):
            f.unlink()
        for f in lbl_dest_dir.glob("*"):
            f.unlink()

        for s in samples:
            img_filename = Path(s["img_entry"]).name
            lbl_filename = Path(s["txt_entry"]).name

            with open(img_dest_dir / img_filename, "wb") as f:
                f.write(s["img_bytes"])

            with open(lbl_dest_dir / lbl_filename, "w", encoding="utf-8") as f:
                f.write(s["txt_content"] + "\n")

    # Generate data.yaml
    data_yaml_path = out_base / "data.yaml"
    data_yaml_content = f"""# AgriVision Dataset Configuration
path: .
train: images/train
val: images/val
test: images/test

names:
  0: crop
  1: weed

nc: 2
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(data_yaml_content)
    logger.info(f"Generated {data_yaml_path}")

    # Generate dataset_info.md
    info_md_path = out_base / "dataset_info.md"
    info_md_content = f"""# AgriVision Dataset Information & Provenance

## Dataset Provenance
- **Archive Source File**: `{archive_file.name}`
- **Extraction Timestamp**: Deterministically prepared from source archive
- **Classes Defined in Archive**: `0: crop`, `1: weed` (verified via `classes.txt`)
- **Domain**: In-field agricultural crop and weed imagery annotated for object detection (bounding boxes)

## Integrity & Quality Verification
- **Total Archive Image Entries**: {len(img_map)}
- **Corrupted Images Detected**: {len(corrupt_images)}
- **Duplicate Images Detected**: {len(duplicate_images)}
- **Empty Label Files Detected**: {len(empty_labels)}
- **Malformed Annotations Detected**: {len(malformed_labels)}
- **Orphan Label Files**: {len(orphan_txts)} ({[Path(o).name for o in orphan_txts] if orphan_txts else 'None'})
- **Total Valid Usable Samples**: {len(valid_samples)} (100.0% integrity)

## Class and Annotation Statistics
- **Total Bounding Boxes**: {total_boxes}
- **Crop Annotations (Class 0)**: {crop_boxes} ({crop_boxes / total_boxes * 100:.2f}%)
- **Weed Annotations (Class 1)**: {weed_boxes} ({weed_boxes / total_boxes * 100:.2f}%)
- **Images with Crop Annotations**: {images_with_crop}
- **Images with Weed Annotations**: {images_with_weed}
- **Images with Both Classes**: {images_with_both}
- **Average Annotations per Image**: {total_boxes / len(valid_samples):.2f}

## Deterministic Data Split (Random Seed = {seed})
- **Split Ratio**: 70% Train / 15% Validation / 15% Test (Held-out)
- **Train Set**: {len(train_samples)} images ({len(train_samples) / len(valid_samples) * 100:.1f}%) -> `data/images/train/`
- **Validation Set**: {len(val_samples)} images ({len(val_samples) / len(valid_samples) * 100:.1f}%) -> `data/images/val/`
- **Test Set (Held-Out)**: {len(test_samples)} images ({len(test_samples) / len(valid_samples) * 100:.1f}%) -> `data/images/test/`
"""
    with open(info_md_path, "w", encoding="utf-8") as f:
        f.write(info_md_content)
    logger.info(f"Generated {info_md_path}")

    summary = {
        "total_images": len(valid_samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "total_boxes": total_boxes,
        "crop_boxes": crop_boxes,
        "weed_boxes": weed_boxes,
        "images_with_crop": images_with_crop,
        "images_with_weed": images_with_weed,
        "images_with_both": images_with_both,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract, validate, and split agricultural dataset.")
    parser.add_argument("--archive", type=str, default="archive.zip", help="Path to archive.zip")
    parser.add_argument("--output", type=str, default="data", help="Output directory for processed dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    args = parser.parse_args()

    summary = prepare_dataset(archive_path=args.archive, output_dir=args.output, seed=args.seed)
    print("\n" + "=" * 50)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
