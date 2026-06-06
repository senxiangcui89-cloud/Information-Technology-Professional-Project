"""Video / webcam / RTSP stream detection with optional CLAHE."""

import argparse
import time
from pathlib import Path

import cv2

from project.models.detector import Detector, DetectorConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect debris in video stream")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--source", default="0", help="Video file path, or 0 for webcam, or RTSP URL")
    parser.add_argument("--output", default="", help="Output video path (optional)")
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

    # Resolve source
    src = args.source
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_count = 0
    fps_ema = 0.0
    alpha = 0.05

    print("[INFO] Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            annotated, detections = detector.detect_and_annotate(frame)
            elapsed = time.perf_counter() - t0
            fps_ema = alpha * (1 / max(elapsed, 1e-6)) + (1 - alpha) * fps_ema

            # HUD
            cv2.putText(annotated, f"FPS: {fps_ema:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(annotated, f"Objects: {len(detections)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if args.clahe:
                cv2.putText(annotated, "CLAHE ON", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if writer:
                writer.write(annotated)

            cv2.imshow("Floating Debris Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames. Avg FPS: {fps_ema:.1f}")


if __name__ == "__main__":
    main()
