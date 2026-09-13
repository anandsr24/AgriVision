"""
RT-DETR-L Training Script for AgriVision.

Fine-tunes Ultralytics RT-DETR-L on the agricultural crop and weed dataset,
captures real hardware and runtime metadata, exports the best checkpoint to models/best.pt,
and saves outputs/training_metadata.json.
"""

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import psutil
import torch
import ultralytics
from ultralytics import RTDETR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")


def collect_system_metadata() -> dict:
    """Collects actual, non-fabricated hardware and environment metadata."""
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None
    cuda_version = torch.version.cuda if cuda_avail else None

    # CPU details
    cpu_name = platform.processor() or "Unknown CPU"
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)

    return {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": cuda_avail,
        "cuda_version": cuda_version,
        "gpu": gpu_name if gpu_name else "None (CPU training)",
        "cpu": f"{cpu_name} ({cpu_count} logical cores, {ram_gb} GB RAM)",
    }


def train_model(
    data: str = "data/data.yaml",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "auto",
    seed: int = 42,
    patience: int = 8,
    project: str = "runs/train",
    name: str = "agrivision_rtdetr",
    workers: int = 0,
    output_dir: str = "outputs",
    model_save_dir: str = "models",
) -> dict:
    """
    Executes RT-DETR-L fine-tuning with full metadata logging.
    """
    data_path = Path(data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_path}")

    # Determine device
    if device == "auto":
        resolved_device = "0" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    logger.info(f"Targeting device: {resolved_device}")
    logger.info(f"Loading pretrained RT-DETR-L: rtdetr-l.pt")

    model = RTDETR("rtdetr-l.pt")

    # Record system metadata
    sys_meta = collect_system_metadata()
    start_time = time.time()

    logger.info(f"Starting training run: epochs={epochs}, imgsz={imgsz}, batch={batch}, seed={seed}, patience={patience}")

    results = model.train(
    data=str(data_path),
    epochs=epochs,
    imgsz=imgsz,
    batch=batch,
    device=resolved_device,
    seed=seed,
    patience=patience,
    warmup_epochs=1,
    cache=True,
    project=project,
    name=name,
    workers=workers,
    exist_ok=True,
    verbose=True,
)
    elapsed_time = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {elapsed_time} seconds.")

    # Locate saved weights
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / name
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    dest_models_dir = Path(model_save_dir).resolve()
    dest_models_dir.mkdir(parents=True, exist_ok=True)
    target_best_pt = dest_models_dir / "best.pt"

    if best_pt.exists():
        shutil.copy2(best_pt, target_best_pt)
        logger.info(f"Copied best checkpoint to {target_best_pt}")
    elif last_pt.exists():
        shutil.copy2(last_pt, target_best_pt)
        logger.warning(f"best.pt not found, copied last.pt to {target_best_pt}")
    else:
        logger.error("No checkpoint file found in training run output directory.")

    # Build and save training metadata
    outputs_path = Path(output_dir).resolve()
    outputs_path.mkdir(parents=True, exist_ok=True)
    metadata_file = outputs_path / "training_metadata.json"

    training_metadata = {
        "model": "RT-DETR-L",
        "checkpoint": "rtdetr-l.pt",
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "seed": seed,
        "device": resolved_device,
        "os": sys_meta["os"],
        "python_version": sys_meta["python_version"],
        "torch_version": sys_meta["torch_version"],
        "ultralytics_version": sys_meta["ultralytics_version"],
        "cuda_version": sys_meta["cuda_version"],
        "gpu": sys_meta["gpu"],
        "cpu_info": sys_meta["cpu"],
        "training_time_seconds": elapsed_time,
        "save_dir": str(save_dir),
        "best_model_path": str(target_best_pt) if target_best_pt.exists() else None,
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2)
    logger.info(f"Saved training metadata to {metadata_file}")

    return training_metadata


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Ultralytics RT-DETR-L on crop & weed dataset.")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (e.g. cpu, 0, or auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory for runs")
    parser.add_argument("--name", type=str, default="agrivision_rtdetr", help="Experiment name")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 recommended on Windows)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for metadata output")
    parser.add_argument("--model_save_dir", type=str, default="models", help="Directory for best.pt")
    args = parser.parse_args()

    metadata = train_model(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        project=args.project,
        name=args.name,
        workers=args.workers,
        output_dir=args.output_dir,
        model_save_dir=args.model_save_dir,
    )

    print("\n" + "=" * 50)
    print("TRAINING PROCESS COMPLETED")
    print("=" * 50)
    print(json.dumps(metadata, indent=2))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
