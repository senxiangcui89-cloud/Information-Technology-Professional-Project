import zipfile
from pathlib import Path

from app.core.config import UPLOAD_DIR

DATASET_DIR = UPLOAD_DIR / "datasets"


def extract_and_validate(zip_path: str, dataset_name: str) -> dict:
    """Extract ZIP and validate YOLO dataset structure."""
    extract_dir = DATASET_DIR / dataset_name
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    result = _validate_structure(extract_dir)
    result["path"] = str(extract_dir)
    return result


def list_datasets() -> list[dict]:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    datasets = []
    for d in sorted(DATASET_DIR.iterdir()):
        if d.is_dir():
            info = _validate_structure(d)
            datasets.append({"name": d.name, **info})
    return datasets


def _validate_structure(dataset_dir: Path) -> dict:
    train_img = dataset_dir / "images" / "train"
    val_img = dataset_dir / "images" / "val"
    train_lbl = dataset_dir / "labels" / "train"
    val_lbl = dataset_dir / "labels" / "val"

    train_count = len(list(train_img.glob("*"))) if train_img.exists() else 0
    val_count = len(list(val_img.glob("*"))) if val_img.exists() else 0
    train_lbl_count = len(list(train_lbl.glob("*"))) if train_lbl.exists() else 0
    val_lbl_count = len(list(val_lbl.glob("*"))) if val_lbl.exists() else 0

    return {
        "valid": train_count > 0 and val_count > 0,
        "train_images": train_count,
        "val_images": val_count,
        "train_labels": train_lbl_count,
        "val_labels": val_lbl_count,
        "structure_ok": train_img.exists() and val_img.exists(),
    }
