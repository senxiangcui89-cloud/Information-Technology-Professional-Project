"""Dataset preparation for YOLO-format object detection.

Handles:
  - Converting Pascal VOC XML annotations to YOLO format.
  - Train/validation split with stratified sampling.
  - Auto-detecting class names from dataset annotations.
  - Generating the dataset.yaml descriptor that ultralytics expects.
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml
from pydantic import BaseModel, Field
from tqdm import tqdm


# ----------------------------------------------------------------
# Config model
# ----------------------------------------------------------------

class DatasetConfig(BaseModel):
    name: str = "FloW-Dataset"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_split: float = Field(default=0.80, ge=0.5, le=0.95)
    val_split: float = Field(default=0.20, ge=0.05, le=0.5)
    seed: int = 42
    class_names: list[str] = Field(default=["bottle"])


# ----------------------------------------------------------------
# VOC XML parsing
# ----------------------------------------------------------------

def parse_voc_xml(xml_path: str | Path) -> list[dict]:
    """Parse a Pascal VOC annotation XML file.

    Returns a list of object dicts, each with keys: name, xmin, ymin, xmax, ymax.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text) if size is not None else 0
    img_h = int(size.find("height").text) if size is not None else 0

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        objects.append({
            "name": name,
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
            "img_w": img_w, "img_h": img_h,
        })
    return objects


def collect_voc_class_names(xml_dir: str | Path) -> list[str]:
    """Scan all XML files in a directory and return a sorted list of unique class names."""
    names_set: set[str] = set()
    xml_dir = Path(xml_dir)
    for xml_path in xml_dir.glob("*.xml"):
        tree = ET.parse(str(xml_path))
        for obj in tree.getroot().findall("object"):
            names_set.add(obj.find("name").text)
    return sorted(names_set)


# ----------------------------------------------------------------
# YOLO-format conversion helpers
# ----------------------------------------------------------------

def bbox_xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
) -> tuple[float, float, float, float]:
    """Convert [x1,y1,x2,y2] absolute coords → YOLO normalized [xc,yc,w,h]."""
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h


def write_yolo_label(
    save_path: Path,
    class_id: int,
    xc: float, yc: float, w: float, h: float,
) -> None:
    """Append a YOLO-format line: class_id xc yc w h"""
    with open(save_path, "a") as f:
        f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


# ----------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------

def prepare_flow_dataset(config: DatasetConfig | None = None) -> str:
    """Convert a raw dataset to YOLO format with train/val split.

    Expects one of these layouts under raw_dir/:

        Layout A (YOLO-ready, no conversion needed):
            raw/
            ├── images/
            │   ├── train/
            │   └── val/
            └── labels/
                ├── train/
                └── val/

        Layout B (images + one .txt label folder):
            raw/
            ├── images/          # all images
            └── labels/          # .txt per image (YOLO format)

        Layout C (VOC XML, pre-split into training/test):
            raw/
            ├── training/
            │   ├── images/      # .jpg/.png files
            │   └── annotations/ # .xml files (Pascal VOC)
            └── test/
                ├── images/
                └── annotations/

    Returns:
        Path to the generated dataset.yaml.
    """
    if config is None:
        config = DatasetConfig()

    random.seed(config.seed)
    raw_dir = Path(config.raw_dir)

    # --- Detect layout ---
    layout_a = (raw_dir / "images" / "train").exists()
    layout_b = (raw_dir / "images").exists() and (raw_dir / "labels").exists()
    layout_c = (
        (raw_dir / "training" / "images").exists()
        and (raw_dir / "training" / "annotations").exists()
        and (raw_dir / "test" / "images").exists()
        and (raw_dir / "test" / "annotations").exists()
    )

    if layout_a:
        print("[INFO] Detected Layout A: YOLO-ready — copying directly.")
        _copy_yolo_structure(raw_dir, Path(config.processed_dir))
    elif layout_c:
        print("[INFO] Detected Layout C: VOC XML with training/test split.")
        class_names = _discover_class_names(raw_dir, config)
        print(f"[INFO] Auto-detected {len(class_names)} classes: {class_names}")
        config.class_names = class_names
        _convert_voc_to_yolo(raw_dir, Path(config.processed_dir), class_names)
    elif layout_b:
        print("[INFO] Detected Layout B: splitting images + labels.")
        _split_and_copy(raw_dir / "images", raw_dir / "labels", Path(config.processed_dir), config)
    else:
        raise FileNotFoundError(
            f"Unrecognized dataset layout under {raw_dir}. "
            f"Expected layout A, B, or C (see docstring)."
        )

    # --- Generate dataset.yaml ---
    yaml_path = _write_dataset_yaml(Path(config.processed_dir), config)
    print(f"[DONE] dataset.yaml written to {yaml_path}")
    return str(yaml_path)


# ----------------------------------------------------------------
# VOC → YOLO conversion (Layout C)
# ----------------------------------------------------------------

def _discover_class_names(raw_dir: Path, config: DatasetConfig) -> list[str]:
    """Scan all XML annotations to discover class names.

    Checks training/annotations/ and test/annotations/ if present,
    falls back to scanning raw_dir directly.
    """
    all_names: set[str] = set()
    for candidate in [
        raw_dir / "training" / "annotations",
        raw_dir / "test" / "annotations",
    ]:
        if candidate.exists():
            all_names.update(collect_voc_class_names(candidate))

    if all_names:
        return sorted(all_names)

    # Fallback: scan raw_dir directly for any .xml files
    xml_files = list(raw_dir.rglob("*.xml"))
    if xml_files:
        for xml_path in xml_files:
            tree = ET.parse(str(xml_path))
            for obj in tree.getroot().findall("object"):
                all_names.add(obj.find("name").text)
        return sorted(all_names)

    # Nothing found — keep the configured default
    return config.class_names


def _convert_voc_to_yolo(
    raw_dir: Path, processed_dir: Path, class_names: list[str],
) -> None:
    """Convert VOC XML dataset with training/test split to YOLO format.

    training/ → train, test/ → val
    """
    name_to_id = {name: i for i, name in enumerate(class_names)}

    split_map = [
        ("training", "train"),
        ("test", "val"),
    ]

    for src_split, dst_split in split_map:
        img_src = raw_dir / src_split / "images"
        ann_src = raw_dir / src_split / "annotations"
        img_dst = processed_dir / "images" / dst_split
        lbl_dst = processed_dir / "labels" / dst_split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = sorted(
            p for p in img_src.iterdir() if p.suffix.lower() in extensions
        )

        for img_path in tqdm(image_files, desc=f"Converting {src_split} → {dst_split}"):
            shutil.copy2(img_path, img_dst / img_path.name)

            xml_path = ann_src / f"{img_path.stem}.xml"
            lbl_path = lbl_dst / f"{img_path.stem}.txt"

            if xml_path.exists():
                objects = parse_voc_xml(xml_path)
                for obj in objects:
                    xc, yc, w, h = bbox_xyxy_to_yolo(
                        obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                        obj["img_w"], obj["img_h"],
                    )
                    cls_id = name_to_id.get(obj["name"], 0)
                    write_yolo_label(lbl_path, cls_id, xc, yc, w, h)
            else:
                # Create empty label file for images with no annotations
                lbl_path.touch()
                tqdm.write(f"[WARN] No XML for {img_path.name}, created empty .txt")

        # Verify alignment
        img_names = {p.stem for p in img_dst.iterdir() if p.suffix.lower() in extensions}
        lbl_names = {p.stem for p in lbl_dst.iterdir() if p.suffix == ".txt"}
        for name in img_names - lbl_names:
            (lbl_dst / f"{name}.txt").touch()
            tqdm.write(f"[WARN] No label for {name}, created empty .txt")


# ----------------------------------------------------------------
# Internal helpers (Layout A / Layout B)
# ----------------------------------------------------------------

def _copy_yolo_structure(raw: Path, processed: Path) -> None:
    """Copy a YOLO-ready dataset directly."""
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            src = raw / kind / split
            dst = processed / kind / split
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                for f in tqdm(list(src.iterdir()), desc=f"Copying {kind}/{split}"):
                    if f.is_file():
                        shutil.copy2(f, dst / f.name)


def _split_and_copy(
    img_dir: Path, label_dir: Path, processed: Path, config: DatasetConfig,
) -> None:
    """Split images/labels into train/val, then copy."""
    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    random.shuffle(image_files)

    n_train = int(len(image_files) * config.train_split)
    train_files = image_files[:n_train]
    val_files = image_files[n_train:]

    for split, files in [("train", train_files), ("val", val_files)]:
        img_out = processed / "images" / split
        lbl_out = processed / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(files, desc=f"Preparing {split}"):
            shutil.copy2(img_path, img_out / img_path.name)

            label_name = img_path.stem + ".txt"
            label_path = label_dir / label_name
            if label_path.exists():
                shutil.copy2(label_path, lbl_out / label_name)
            else:
                (lbl_out / label_name).touch()

        # Verify images and labels are aligned
        img_names = {p.stem for p in img_out.iterdir() if p.suffix in {".jpg", ".jpeg", ".png"}}
        lbl_names = {p.stem for p in lbl_out.iterdir() if p.suffix == ".txt"}
        missing_labels = img_names - lbl_names
        for name in missing_labels:
            (lbl_out / f"{name}.txt").touch()
            tqdm.write(f"[WARN] No label for {name}, created empty .txt")


def _write_dataset_yaml(processed_dir: Path, config: DatasetConfig) -> Path:
    """Write data/dataset.yaml for ultralytics."""
    yaml_content = {
        "path": str(processed_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(config.class_names)},
    }
    yaml_path = Path("data") / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return yaml_path


# ----------------------------------------------------------------
# Convenience: run end-to-end CLAHE + dataset prep
# ----------------------------------------------------------------

def prepare_pipeline(
    raw_dir: str = "data/raw",
    use_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
) -> str:
    """One-shot: run CLAHE (optional) then convert to YOLO format.

    Returns path to dataset.yaml.
    """
    if use_clahe:
        from project.data.preprocess import CLAHEConfig, process_directory

        print("[STEP 1/2] Running CLAHE preprocessing...")
        clahe_cfg = CLAHEConfig(clip_limit=clahe_clip_limit)
        clahe_out = Path("data") / "clahe_intermediate"
        process_directory(raw_dir, clahe_out, config=clahe_cfg)

        print("[STEP 2/2] Converting to YOLO format...")
        config = DatasetConfig(raw_dir=str(clahe_out))
    else:
        config = DatasetConfig(raw_dir=raw_dir)

    return prepare_flow_dataset(config)
