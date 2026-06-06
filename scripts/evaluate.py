"""Evaluate all trained models and generate comparison metrics."""

import json
from pathlib import Path

from ultralytics import YOLO

MODELS = {
    "YOLO11n_raw": "experiments/exp_yolo11n_raw/train/weights/best.pt",
    "YOLO11s_raw": "experiments/exp_yolo11s_raw/train/weights/best.pt",
    "YOLO11n_clahe": "experiments/exp_yolo11n_clahe/train/weights/best.pt",
    "YOLO11s_clahe": "experiments/exp_yolo11s_clahe/train/weights/best.pt",
}

RESULTS_FILE = Path("experiments/eval_results.json")


def main():
    results = {}
    for name, weights in MODELS.items():
        if not Path(weights).exists():
            print(f"SKIP {name}: {weights} not found")
            continue

        model = YOLO(weights)
        metrics = model.val(data="data/dataset.yaml", split="val", verbose=False)

        results[name] = {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50-95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            "fps": round(1000 / metrics.speed["inference"], 1) if metrics.speed.get("inference") else None,
            "inference_ms": round(metrics.speed.get("inference", 0), 1),
        }
        print(f"{name}: mAP50={results[name]['mAP50']:.4f}  FPS={results[name]['fps']}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, RESULTS_FILE.open("w"), indent=2)

    print("\n--- Comparison ---")
    hdr = "%-16s %8s %10s %8s %8s %8s %10s" % ("Model", "mAP50", "mAP50-95", "P", "R", "FPS", "Infer(ms)")
    print(hdr)
    print("-" * len(hdr))
    for name, m in results.items():
        print(
            "%-16s %8.4f %10.4f %8.4f %8.4f %8s %10.1f"
            % (
                name,
                m["mAP50"],
                m["mAP50-95"],
                m["precision"],
                m["recall"],
                str(m["fps"]) if m["fps"] else "N/A",
                m["inference_ms"],
            )
        )


if __name__ == "__main__":
    main()
