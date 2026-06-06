import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.detector import AVAILABLE_MODELS
from ultralytics import YOLO


def run_evaluation(model_name: str, dataset_path: str, task_id: int) -> dict:
    """Run model.val() on a dataset and return metrics."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    # Create temp dataset YAML
    names = _detect_class_names(dataset_path)
    yaml_path = Path(dataset_path) / "dataset.yaml"
    yaml_content = {
        "path": str(Path(dataset_path).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(names)} if names else {0: "object"},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)

    model = YOLO(AVAILABLE_MODELS[model_name]["path"])
    metrics = model.val(data=str(yaml_path), split="val", verbose=False)

    result = {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "fps": round(1000 / metrics.speed["inference"], 1) if metrics.speed.get("inference") else None,
        "inference_ms": round(metrics.speed.get("inference", 0), 1),
    }

    # Save result JSON alongside dataset
    result_path = Path(dataset_path) / f"eval_{model_name}.json"
    json.dump(result, open(result_path, "w"), indent=2)

    yaml_path.unlink(missing_ok=True)
    return result


def _detect_class_names(dataset_path: str) -> list[str]:
    """Try to read class names from dataset."""
    p = Path(dataset_path)
    # Check for data.yaml or dataset.yaml
    for yf in ["data.yaml", "dataset.yaml"]:
        yaml_path = p / yf
        if yaml_path.exists():
            try:
                with open(yaml_path) as f:
                    data = yaml.load(f, Loader=yaml.SafeLoader)
                    if data and "names" in data:
                        if isinstance(data["names"], list):
                            return data["names"]
                        if isinstance(data["names"], dict):
                            return list(data["names"].values())
            except Exception:
                pass

    # Default: try to read from labels
    lbl_dir = p / "labels" / "val"
    if not lbl_dir.exists():
        lbl_dir = p / "labels" / "train"
    if lbl_dir.exists():
        # YOLO format: class_id is first column
        return ["bottle"]  # Default for this project
    return ["bottle"]
