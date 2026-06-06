#!/usr/bin/env python3
"""Complete experimental pipeline for the floating debris detection project.

Usage:
    # Step 1 — Prepare dataset (no CLAHE)
    python scripts/run_pipeline.py --step prep

    # Step 2 — Prepare CLAHE dataset
    python scripts/run_pipeline.py --step prep --clahe

    # Step 3 — Train all models
    python scripts/run_pipeline.py --step train

    # Step 4 — Evaluate & compare
    python scripts/run_pipeline.py --step eval

    # All steps (end-to-end)
    python scripts/run_pipeline.py --step all
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def step_prepare(clahe: bool = False) -> None:
    """Download/prepare dataset, optionally with CLAHE."""
    from project.data.dataset_prep import prepare_pipeline

    print("=" * 60)
    print(f"STEP: Dataset preparation {'(CLAHE enabled)' if clahe else ''}")
    print("=" * 60)

    yaml_path = prepare_pipeline(
        raw_dir=str(PROJECT_ROOT / "dataset" / "FloW_IMG"),
        use_clahe=clahe,
    )
    print(f"dataset.yaml → {yaml_path}")


def step_train() -> None:
    """Train all models defined in default.yaml."""
    from project.train.train import run_all_experiments

    print("=" * 60)
    print("STEP: Training all experiments")
    print("=" * 60)

    results = run_all_experiments(
        config_path=str(PROJECT_ROOT / "project" / "config" / "default.yaml"),
        experiments_dir=str(PROJECT_ROOT / "experiments"),
    )

    print("\nTraining summary:")
    for r in results:
        print(f"  {r['exp_name']}: mAP50={r['best_mAP50']:.4f}, time={r['train_time_s']:.1f}s")


def step_evaluate() -> None:
    """Evaluate and compare all trained models."""
    from project.eval.evaluate import batch_compare

    print("=" * 60)
    print("STEP: Evaluation & comparison")
    print("=" * 60)

    comparisons = batch_compare(
        experiments_dir=str(PROJECT_ROOT / "experiments"),
        data_yaml=str(PROJECT_ROOT / "data" / "dataset.yaml"),
    )

    if not comparisons:
        print("No experiment pairs found. Run training first.")
        return

    print(f"\nCompared {len(comparisons)} model pairs.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Floating Debris Detection Pipeline")
    parser.add_argument(
        "--step",
        required=True,
        choices=["prep", "train", "eval", "all"],
        help="Pipeline step to run",
    )
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE during prep step")
    args = parser.parse_args()

    if args.step in ("prep", "all"):
        step_prepare(clahe=args.clahe)

    if args.step in ("train", "all"):
        step_train()

    if args.step in ("eval", "all"):
        step_evaluate()

    print("\n[DONE] Pipeline completed.")


if __name__ == "__main__":
    main()
