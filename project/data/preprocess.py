"""CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing.

Applies CLAHE in LAB color space to enhance local contrast without amplifying
noise — critical for detecting small objects in glare/reflection-prone water scenes.

Reference:
    Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization.
    Graphics Gems IV, 474-485.
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Validate inputs with pydantic
from pydantic import BaseModel, Field, model_validator


class CLAHEConfig(BaseModel):
    clip_limit: float = Field(default=2.0, ge=0.1, le=40.0)
    tile_grid_size: tuple[int, int] = Field(default=(8, 8))
    color_space: str = Field(default="LAB")

    @model_validator(mode="after")
    def _check_color_space(self) -> "CLAHEConfig":
        if self.color_space not in ("LAB", "YUV", "BGR"):
            raise ValueError(f"Unsupported color_space: {self.color_space}")
        return self


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    color_space: str = "LAB",
) -> np.ndarray:
    """Apply CLAHE to a BGR image, enhancing the luminance channel only.

    Args:
        image: Input BGR image (H, W, 3).
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid size for adaptive histogram equalization.
        color_space: 'LAB' (recommended), 'YUV', or 'BGR' (per-channel).

    Returns:
        CLAHE-enhanced BGR image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if color_space == "LAB":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    if color_space == "YUV":
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv)
        y_eq = clahe.apply(y)
        yuv_eq = cv2.merge((y_eq, u, v))
        return cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2BGR)

    # BGR fallback — apply CLAHE to each channel independently
    channels = cv2.split(image)
    eq_channels = [clahe.apply(ch) for ch in channels]
    return cv2.merge(eq_channels)


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    config: CLAHEConfig | None = None,
    save_comparison: bool = False,
    comparison_dir: str | Path | None = None,
) -> tuple[int, Path]:
    """Batch-apply CLAHE to all images in a directory.

    Args:
        input_dir: Directory containing input images.
        output_dir: Directory to save enhanced images.
        config: CLAHE parameters.
        save_comparison: If True, save side-by-side comparison images.
        comparison_dir: Directory for comparison images.

    Returns:
        (count of processed images, output_dir path).
    """
    if config is None:
        config = CLAHEConfig()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in extensions
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    comp_dir = None
    if save_comparison:
        comp_dir = Path(comparison_dir or output_dir / "_comparisons")
        comp_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_paths, desc="CLAHE processing", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            tqdm.write(f"[WARN] Cannot read {img_path.name}, skipping")
            continue

        enhanced = apply_clahe(
            img,
            clip_limit=config.clip_limit,
            tile_grid_size=config.tile_grid_size,
            color_space=config.color_space,
        )

        # Preserve subdirectory structure
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), enhanced)

        if comp_dir:
            comp = _make_comparison(img, enhanced)
            comp_path = comp_dir / f"cmp_{img_path.stem}.jpg"
            cv2.imwrite(str(comp_path), comp)

    return len(image_paths), output_dir


def _make_comparison(original: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """Horizontal concatenation: original | enhanced with labels."""
    h, w = original.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    label_h = 40
    canvas = np.ones((h + label_h, w * 2, 3), dtype=np.uint8) * 255
    canvas[label_h:, :w] = original
    canvas[label_h:, w:] = enhanced
    cv2.putText(canvas, "Original", (10, 28), font, 0.8, (0, 0, 0), 2)
    cv2.putText(canvas, "CLAHE", (w + 10, 28), font, 0.8, (0, 0, 0), 2)
    return canvas
