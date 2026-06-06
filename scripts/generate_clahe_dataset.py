"""Generate CLAHE-preprocessed copy of the dataset."""

import shutil
from pathlib import Path

import cv2
from tqdm import tqdm

SRC = Path("data/processed")
DST = Path("data/clahe_processed")
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for split in ("train", "val"):
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_dir = SRC / "images" / split
    for img_path in tqdm(list(img_dir.glob("*.jpg")), desc=f"CLAHE {split}"):
        img = cv2.imread(str(img_path))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = CLAHE.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        out = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(DST / "images" / split / img_path.name), out)

    # Copy labels unchanged
    for lbl in (SRC / "labels" / split).glob("*.txt"):
        shutil.copy2(lbl, DST / "labels" / split / lbl.name)

print("Done. CLAHE dataset at", DST)
