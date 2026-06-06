"""Evaluation and comparison module.

Computes detection metrics (mAP, Precision, Recall) and inference speed (FPS)
for trained models, and produces tabular + visual comparisons between
raw-image and CLAHE-enhanced models.
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field

from project.utils.metrics_plot import plot_fps_comparison, plot_metric_comparison


# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------

class EvalConfig(BaseModel):
    data_yaml: str = Field(default="data/dataset.yaml")
    conf_threshold: float = Field(default=0.25, ge=0, le=1)
    iou_threshold: float = Field(default=0.45, ge=0, le=1)
    benchmark_iters: int = Field(default=100, ge=10)
    device: str = Field(default="cuda")
    save_plots: bool = Field(default=True)


# ----------------------------------------------------------------
# Metric evaluation (mAP, Precision, Recall)
# ----------------------------------------------------------------

def evaluate_model(
    weights_path: str | Path,
    data_yaml: str = "data/dataset.yaml",
    config: EvalConfig | None = None,
) -> dict[str, float]:
    """Run ultralytics validation mode and return detection metrics.

    Args:
        weights_path: Path to trained .pt weights.
        data_yaml: Path to dataset YAML descriptor.
        config: Evaluation config.

    Returns:
        Dict with mAP50, mAP50_95, precision, recall, f1.
    """
    from ultralytics import YOLO

    if config is None:
        config = EvalConfig()

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=data_yaml,
        conf=config.conf_threshold,
        iou=config.iou_threshold,
        device=config.device,
        split="val",
    )

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


# ----------------------------------------------------------------
# Inference speed benchmark (FPS)
# ----------------------------------------------------------------

def benchmark_fps(
    weights_path: str | Path,
    imgsz: int = 640,
    device: str = "cuda",
    n_iters: int = 100,
    warmup_iters: int = 20,
) -> dict[str, float]:
    """Measure inference throughput in FPS.

    Args:
        weights_path: Path to .pt weights.
        imgsz: Input image size.
        device: 'cuda' or 'cpu'.
        n_iters: Benchmark iterations.
        warmup_iters: Warm-up iterations (excluded from timing).

    Returns:
        Dict with fps, mean_latency_ms, std_latency_ms.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    # Generate random input tensor
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device_obj)

    latencies: list[float] = []

    for i in range(warmup_iters + n_iters):
        torch.cuda.synchronize() if device_obj.type == "cuda" else None
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.predict(dummy, verbose=False)
        if device_obj.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if i >= warmup_iters:
            latencies.append(elapsed * 1000)  # ms

    mean_ms = float(np.mean(latencies))
    std_ms = float(np.std(latencies))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "fps": round(fps, 1),
        "mean_latency_ms": round(mean_ms, 2),
        "std_latency_ms": round(std_ms, 2),
    }


# ----------------------------------------------------------------
# Head-to-head comparison: Raw vs CLAHE
# ----------------------------------------------------------------

def compare_raw_vs_clahe(
    raw_weights: str | Path,
    clahe_weights: str | Path,
    data_yaml: str = "data/dataset.yaml",
    output_dir: str | Path = "experiments/comparison",
    config: EvalConfig | None = None,
) -> dict[str, Any]:
    """Evaluate both models and produce a comparison report.

    Args:
        raw_weights: Path to the raw-image trained model.
        clahe_weights: Path to the CLAHE-enhanced trained model.
        data_yaml: Dataset YAML path.
        output_dir: Where to save plots and report.
        config: Evaluation config.

    Returns:
        Comparison dictionary suitable for inclusion in a report.
    """
    if config is None:
        config = EvalConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[EVAL] Computing detection metrics...")
    raw_metrics = evaluate_model(raw_weights, data_yaml, config)
    clahe_metrics = evaluate_model(clahe_weights, data_yaml, config)

    print("[EVAL] Benchmarking inference speed...")
    raw_fps = benchmark_fps(raw_weights, device=config.device, n_iters=config.benchmark_iters)
    clahe_fps = benchmark_fps(clahe_weights, device=config.device, n_iters=config.benchmark_iters)

    # Build comparison
    deltas = {}
    for k in raw_metrics:
        deltas[k] = round(clahe_metrics[k] - raw_metrics[k], 4)

    comparison = {
        "raw": {**raw_metrics, **raw_fps},
        "clahe": {**clahe_metrics, **clahe_fps},
        "delta": deltas,
        "fps_delta": round(clahe_fps["fps"] - raw_fps["fps"], 1),
    }

    # Save report
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Generate plots
    if config.save_plots:
        print("[EVAL] Generating comparison plots...")
        plot_metric_comparison(
            raw_metrics, clahe_metrics,
            save_path=output_dir / "metrics_comparison.png",
        )
        plot_fps_comparison(
            {"Raw": raw_fps["fps"], "CLAHE": clahe_fps["fps"]},
            save_path=output_dir / "fps_comparison.png",
        )

    # Pretty-print summary
    print("\n" + "=" * 60)
    print("COMPARISON REPORT: Raw vs CLAHE")
    print("=" * 60)
    header = f"{'Metric':<16} {'Raw':>10} {'CLAHE':>10} {'Delta':>10}"
    print(header)
    print("-" * 46)
    for k in raw_metrics:
        print(f"{k:<16} {raw_metrics[k]:>10.4f} {clahe_metrics[k]:>10.4f} {deltas[k]:>+10.4f}")
    print(f"{'FPS':<16} {raw_fps['fps']:>10.1f} {clahe_fps['fps']:>10.1f} {comparison['fps_delta']:>+10.1f}")
    print("-" * 46)
    print(f"Report saved → {report_path}")

    return comparison


# ----------------------------------------------------------------
# Batch comparison across all experiments
# ----------------------------------------------------------------

def batch_compare(
    experiments_dir: str | Path = "experiments",
    data_yaml: str = "data/dataset.yaml",
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Compare all experiment pairs (raw vs clahe) found in experiments_dir."""
    experiments_dir = Path(experiments_dir)
    comparisons = []

    # Find raw/clahe pairs by naming convention: exp_<model>_raw / exp_<model>_clahe
    raw_dirs = sorted(experiments_dir.glob("exp_*_raw"))
    for raw_dir in raw_dirs:
        clahe_dir = Path(str(raw_dir).replace("_raw", "_clahe"))
        if not clahe_dir.exists():
            continue

        raw_pt = raw_dir / "best.pt"
        clahe_pt = clahe_dir / "best.pt"
        if not raw_pt.exists() or not clahe_pt.exists():
            continue

        print(f"\n{'=' * 50}")
        print(f"Comparing: {raw_dir.name}  vs  {clahe_dir.name}")
        print(f"{'=' * 50}")

        comparison = compare_raw_vs_clahe(
            raw_pt, clahe_pt, data_yaml,
            output_dir=experiments_dir / f"comparison_{raw_dir.name.replace('_raw', '')}",
        )
        comparisons.append(comparison)

    return comparisons
