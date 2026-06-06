"""Single-image detection with optional CLAHE preprocessing."""

import argparse
from pathlib import Path

import cv2

from project.models.detector import Detector, DetectorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect debris in a single image")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--source", required=True, help="Path to input image")
    parser.add_argument("--output", default="output.jpg", help="Output path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    args = parser.parse_args()

    config = DetectorConfig(
        weights=args.weights,
        conf=args.conf,
        device=args.device,
        clahe_enabled=args.clahe,
    )
    detector = Detector(config)

    img = cv2.imread(args.source)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.source}")

    annotated, detections = detector.detect_and_annotate(img)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, annotated)

    print(f"Found {len(detections)} objects → {args.output}")
    for d in detections:
        print(f"  {d.class_name}: {d.confidence:.2%} @ {d.bbox_xyxy}")


if __name__ == "__main__":
    main()
