"""Unified detector interface for all YOLO variants."""

from pathlib import Path

import cv2
import numpy as np
import torch
from pydantic import BaseModel, Field

from project.data.preprocess import apply_clahe


class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2


class DetectorConfig(BaseModel):
    weights: str
    imgsz: int = Field(default=640, ge=320)
    conf: float = Field(default=0.25, ge=0, le=1)
    iou: float = Field(default=0.45, ge=0, le=1)
    device: str = "cuda"
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    class_names: dict[int, str] = Field(default_factory=dict)


class Detector:
    """Wrapper around ultralytics YOLO with optional CLAHE preprocessing."""

    def __init__(self, config: DetectorConfig):
        from ultralytics import YOLO

        self.config = config
        self.model = YOLO(config.weights)
        self._warmup()

    def _warmup(self) -> None:
        """Run a single dummy inference to warm JIT / GPU."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy, verbose=False)

    def detect(self, image: np.ndarray, verbose: bool = False) -> list[DetectionResult]:
        """Run detection on a BGR image.

        Args:
            image: BGR numpy array (H, W, 3).
            verbose: If True, print progress.

        Returns:
            List of DetectionResult.
        """
        if self.config.clahe_enabled:
            image = apply_clahe(
                image,
                clip_limit=self.config.clahe_clip_limit,
                tile_grid_size=(8, 8),
                color_space="LAB",
            )

        results = self.model.predict(
            image,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            verbose=verbose,
        )

        detections: list[DetectionResult] = []
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confs, clss):
                detections.append(DetectionResult(
                    class_id=int(cls_id),
                    class_name=self.config.class_names.get(int(cls_id), f"class_{cls_id}"),
                    confidence=round(float(conf), 4),
                    bbox_xyxy=tuple(float(v) for v in box),
                ))
        return detections

    def detect_and_annotate(
        self, image: np.ndarray, verbose: bool = False,
    ) -> tuple[np.ndarray, list[DetectionResult]]:
        """Run detection and draw bounding boxes on a copy of the image."""
        dets = self.detect(image, verbose=verbose)
        annotated = image.copy()
        for d in dets:
            x1, y1, x2, y2 = map(int, d.bbox_xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated, dets
