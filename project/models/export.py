"""Model export to ONNX / TensorRT / OpenVINO for deployment."""

import argparse
from pathlib import Path

from pydantic import BaseModel, Field


class ExportConfig(BaseModel):
    weights: str
    format: str = Field(default="onnx")  # onnx | engine | openvino | tflite
    imgsz: int = Field(default=640)
    half: bool = Field(default=False)
    device: str = Field(default="cpu")
    output_dir: str = Field(default="weights/exported")


def export_model(config: ExportConfig) -> str:
    """Export a trained YOLO model to deployment format.

    Args:
        config: Export configuration.

    Returns:
        Path to the exported model file.
    """
    from ultralytics import YOLO

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(config.weights)
    exported_path = model.export(
        format=config.format,
        imgsz=config.imgsz,
        half=config.half,
        device=config.device,
    )
    return str(exported_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO model for deployment")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--format", default="onnx", choices=["onnx", "engine", "openvino", "tflite"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="FP16 export")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = ExportConfig(
        weights=args.weights,
        format=args.format,
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
    )
    path = export_model(config)
    print(f"Exported model → {path}")


if __name__ == "__main__":
    main()
