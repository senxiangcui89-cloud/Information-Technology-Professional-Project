"""Re-train experiments with consistent lr0=0.001."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from project.train.train import train_experiment, ExperimentConfig, TrainingConfig

    training = TrainingConfig(
        epochs=100,
        batch_size=16,
        imgsz=640,
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=50,
        device="0",
        amp=True,
        cos_lr=True,
        close_mosaic=10,
    )

    experiments = [
        ExperimentConfig(
            name="exp_yolo11s_raw",
            description="YOLO11s — raw images (lr0=0.001 fix)",
            model="yolo11s",
            weights="weights/pretrained/yolo11s.pt",
            data_yaml="data/dataset.yaml",
            training=training,
            seed=42,
        ),
        ExperimentConfig(
            name="exp_yolo11n_clahe",
            description="YOLO11n — CLAHE (lr0=0.001 fix)",
            model="yolo11n",
            weights="weights/pretrained/yolo11n.pt",
            data_yaml="data/clahe_dataset.yaml",
            training=training,
            seed=42,
        ),
        ExperimentConfig(
            name="exp_yolo11s_clahe",
            description="YOLO11s — CLAHE (lr0=0.001 fix)",
            model="yolo11s",
            weights="weights/pretrained/yolo11s.pt",
            data_yaml="data/clahe_dataset.yaml",
            training=training,
            seed=42,
        ),
    ]

    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Training: {exp.name} — {exp.description}")
        print(f"Model: {exp.model}  Data: {exp.data_yaml}  lr0: {exp.training.lr0}")
        print(f"{'='*60}")
        result = train_experiment(exp, experiments_dir=str(PROJECT_ROOT / "experiments"))
        print(f"Result: mAP50={result['best_mAP50']:.4f}  mAP50-95={result['best_mAP50_95']:.4f}")

    print("\n[DONE] All experiments re-trained with lr0=0.001")


if __name__ == "__main__":
    main()
