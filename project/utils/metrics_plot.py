"""Visualization utilities for detection results and metric comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_pr_curve(
    precisions: list[float],
    recalls: list[float],
    save_path: str | Path,
    model_name: str = "",
) -> str:
    """Plot Precision-Recall curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, "b-", linewidth=2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"PR Curve — {model_name}", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_metric_comparison(
    raw_metrics: dict[str, float],
    clahe_metrics: dict[str, float],
    save_path: str | Path,
    metric_names: list[str] | None = None,
) -> str:
    """Side-by-side bar chart comparing raw vs CLAHE model metrics."""
    if metric_names is None:
        metric_names = ["mAP50", "mAP50-95", "precision", "recall"]

    x = np.arange(len(metric_names))
    width = 0.35

    raw_values = [raw_metrics.get(m, 0) for m in metric_names]
    clahe_values = [clahe_metrics.get(m, 0) for m in metric_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, raw_values, width, label="Raw", color="#3498db")
    bars2 = ax.bar(x + width / 2, clahe_values, width, label="CLAHE", color="#e74c3c")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Raw vs CLAHE — Detection Performance", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return str(save_path)


def plot_fps_comparison(
    fps_results: dict[str, float],
    save_path: str | Path,
) -> str:
    """Horizontal bar chart comparing FPS across models/configs."""
    names = list(fps_results.keys())
    values = list(fps_results.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3498db" if "raw" in n.lower() else "#e74c3c" for n in names]
    bars = ax.barh(names, values, color=colors)
    ax.set_xlabel("FPS", fontsize=12)
    ax.set_title("Inference Speed Comparison", fontsize=14)

    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    return str(save_path)
