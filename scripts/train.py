#!/usr/bin/env python3
"""Training script for wafer defect detection model.

Supports W&B and MLflow logging, multi-GPU training,
and automatic hyperparameter scheduling.

Usage:
    python scripts/train.py --config configs/training.yaml
"""

import argparse
import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(config_path: str):
    """Run training with configuration."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]

    # Load pretrained model
    model = YOLO(model_cfg.get("pretrained", "yolov8l.pt"))

    # Training arguments
    args = {
        "data": cfg["data"]["train"].replace("train", "data.yaml"),
        "epochs": train_cfg["epochs"],
        "batch": train_cfg["batch_size"],
        "imgsz": model_cfg["img_size"],
        "optimizer": train_cfg["optimizer"],
        "lr0": train_cfg["lr0"],
        "lrf": train_cfg["lrf"],
        "momentum": train_cfg["momentum"],
        "weight_decay": train_cfg["weight_decay"],
        "warmup_epochs": train_cfg["warmup_epochs"],
        "mosaic": aug_cfg["mosaic"],
        "mixup": aug_cfg["mixup"],
        "hsv_h": aug_cfg["hsv_h"],
        "hsv_s": aug_cfg["hsv_s"],
        "hsv_v": aug_cfg["hsv_v"],
        "degrees": aug_cfg["degrees"],
        "translate": aug_cfg["translate"],
        "scale": aug_cfg["scale"],
        "shear": aug_cfg["shear"],
        "flipud": aug_cfg["flipud"],
        "fliplr": aug_cfg["fliplr"],
        "project": cfg["logging"].get("project", "runs/train"),
        "name": cfg["logging"].get("name", "exp"),
    }

    logger.info(f"Starting training for {train_cfg['epochs']} epochs")
    model.train(**args)
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    args = parser.parse_args()
    train(args.config)
