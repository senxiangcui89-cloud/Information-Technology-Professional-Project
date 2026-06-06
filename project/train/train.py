"""Multi-YOLO training pipeline for floating debris detection.

Supports YOLOv8, YOLO11, and YOLO12 trained on raw and CLAHE-enhanced datasets.
Each training run is isolated as an "experiment" with its own config snapshot,
weights, and result logs.
"""

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from project.utils.logger import ExperimentLogger


# ----------------------------------------------------------------
# Configuration models (Pydantic — runtime validation)
# ----------------------------------------------------------------

class TrainingConfig(BaseModel):
    """Training hyperparameters with validation."""
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=1, le=512)
    imgsz: int = Field(default=640, ge=320, le=1920)
    workers: int = Field(default=4, ge=0, le=64)
    optimizer: str = Field(default="AdamW")
    lr0: float = Field(default=0.001, gt=0)
    lrf: float = Field(default=0.01, gt=0)
    momentum: float = Field(default=0.937, ge=0, le=1)
    weight_decay: float = Field(default=0.0005, ge=0)
    warmup_epochs: int = Field(default=3, ge=0)
    warmup_momentum: float = Field(default=0.8, ge=0, le=1)
    warmup_bias_lr: float = Field(default=0.1, gt=0)
    patience: int = Field(default=50, ge=1)
    device: str = Field(default="cuda")
    amp: bool = Field(default=True)
    cos_lr: bool = Field(default=True)
    close_mosaic: int = Field(default=10, ge=0)


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""
    name: str
    description: str = ""
    model: str
    weights: str | None = None          # path to pretrained .pt, or None → auto-download
    data_yaml: str = "data/dataset.yaml"
    clahe_enabled: bool = False
    clahe_overrides: dict[str, Any] = Field(default_factory=dict)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    seed: int = 42


# ----------------------------------------------------------------
# Core training function
# ----------------------------------------------------------------

def train_experiment(
    config: ExperimentConfig,
    experiments_dir: str | Path = "experiments",
    resume: bool = False,
) -> dict[str, Any]:
    """Run a single detection experiment.

    Args:
        config: ExperimentConfig with model, data_yaml, training params.
        experiments_dir: Root directory for experiment outputs.
        resume: If True, pass resume=True to ultralytics.

    Returns:
        Dictionary with keys: model_name, exp_name, best_mAP50, best_mAP50_95,
        precision, recall, train_time_s, results_dir.
    """
    from ultralytics import YOLO

    exp_dir = Path(experiments_dir) / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = ExperimentLogger(exp_dir)

    # Snapshot config for reproducibility
    config_path = exp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False)

    # Resolve model weights
    weights_path = config.weights
    if weights_path is None:
        weights_path = f"{config.model}.pt"   # ultralytics auto-downloads

    logger.log(message="Training started", model=config.model, weights=str(weights_path))

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    model = YOLO(weights_path)
    t0 = time.time()

    results = model.train(
        data=config.data_yaml,
        epochs=config.training.epochs,
        batch=config.training.batch_size,
        imgsz=config.training.imgsz,
        workers=config.training.workers,
        optimizer=config.training.optimizer,
        lr0=config.training.lr0,
        lrf=config.training.lrf,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
        warmup_epochs=config.training.warmup_epochs,
        warmup_momentum=config.training.warmup_momentum,
        warmup_bias_lr=config.training.warmup_bias_lr,
        patience=config.training.patience,
        device=config.training.device,
        amp=config.training.amp,
        cos_lr=config.training.cos_lr,
        close_mosaic=config.training.close_mosaic,
        seed=config.seed,
        project=str(exp_dir),
        name="train",
        exist_ok=True,
        resume=resume,
    )

    train_time = round(time.time() - t0, 1)

    # ----------------------------------------------------------------
    # Collect metrics from the results object
    # ----------------------------------------------------------------
    rd = results.results_dict
    best_map50 = float(rd.get("metrics/mAP50(B)", 0))
    best_map50_95 = float(rd.get("metrics/mAP50-95(B)", 0))
    precision = float(rd.get("metrics/precision(B)", 0))
    recall = float(rd.get("metrics/recall(B)", 0))

    # Copy best.pt to a stable name
    src_best = exp_dir / "train" / "weights" / "best.pt"
    dst_best = exp_dir / "best.pt"
    if src_best.exists():
        shutil.copy2(src_best, dst_best)

    summary = {
        "model_name": config.model,
        "exp_name": config.name,
        "clahe_enabled": config.clahe_enabled,
        "best_mAP50": best_map50,
        "best_mAP50_95": best_map50_95,
        "precision": precision,
        "recall": recall,
        "train_time_s": train_time,
        "results_dir": str(exp_dir),
    }

    logger.finalize(summary)
    print(f"[DONE] {config.name} — mAP50: {best_map50:.4f}, time: {train_time:.1f}s")
    return summary


# ----------------------------------------------------------------
# Batch training — compare models
# ----------------------------------------------------------------

def run_all_experiments(
    config_path: str = "project/config/default.yaml",
    experiments_dir: str = "experiments",
) -> list[dict[str, Any]]:
    """Run all experiments defined in the default config.

    Returns a list of summary dicts (one per experiment).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results = []
    training_base = TrainingConfig(**cfg["training"])

    for model_cfg in cfg["models"]:
        # --- Raw (no CLAHE) ---
        raw_config = ExperimentConfig(
            name=f"exp_{model_cfg['name']}_raw",
            description=f"{model_cfg['name']} — raw images",
            model=model_cfg["name"],
            weights=model_cfg["weights"],
            clahe_enabled=False,
            training=training_base,
            seed=cfg["project"]["seed"],
        )
        results.append(train_experiment(raw_config, experiments_dir))

        # --- CLAHE ---
        clahe_config = ExperimentConfig(
            name=f"exp_{model_cfg['name']}_clahe",
            description=f"{model_cfg['name']} — CLAHE enhanced",
            model=model_cfg["name"],
            weights=model_cfg["weights"],
            clahe_enabled=True,
            clahe_overrides=cfg.get("clahe", {}),
            training=training_base,
            seed=cfg["project"]["seed"],
        )
        results.append(train_experiment(clahe_config, experiments_dir))

    # Write a combined summary
    summary_path = Path(experiments_dir) / "all_results.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] {len(results)} experiments completed → {summary_path}")
    return results
