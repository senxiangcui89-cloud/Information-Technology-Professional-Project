"""End-to-end demo: gradio web UI for interactive debris detection."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from project.models.detector import Detector, DetectorConfig


def build_ui(weights: str, device: str = "cuda") -> None:
    """Launch a Gradio web interface."""
    import gradio as gr

    config = DetectorConfig(weights=weights, device=device)
    detector = Detector(config)

    def detect_fn(image: np.ndarray, conf: float, clahe: bool) -> np.ndarray:
        detector.config.conf = conf
        detector.config.clahe_enabled = clahe
        annotated, detections = detector.detect_and_annotate(image)
        return annotated

    gr.Interface(
        fn=detect_fn,
        inputs=[
            gr.Image(type="numpy", label="Input Image"),
            gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="Confidence Threshold"),
            gr.Checkbox(label="CLAHE Preprocessing"),
        ],
        outputs=gr.Image(type="numpy", label="Detection Results"),
        title="Water Surface Floating Debris Detection",
        description="Upload a water-surface image to detect floating debris.",
        examples=[],
    ).launch(server_name="0.0.0.0", share=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio demo for debris detection")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--device", default="cuda", help="cuda | cpu")
    args = parser.parse_args()
    build_ui(args.weights, args.device)


if __name__ == "__main__":
    main()
