import sys
from pathlib import Path

# Ensure project root on path for ultralytics import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.core.config import MODELS_DIR
from ultralytics import YOLO

AVAILABLE_MODELS = {
    "yolo11n_raw": {
        "path": str(MODELS_DIR / "exp_yolo11n_raw/train/weights/best.pt"),
        "display_name": "YOLO11n (原始数据)",
        "params": "2.6M",
        "mAP50": 0.8383,
        "mAP50_95": 0.4174,
        "fps": None,
    },
    "yolo11s_raw": {
        "path": str(MODELS_DIR / "exp_yolo11s_raw/train/weights/best.pt"),
        "display_name": "YOLO11s (原始数据)",
        "params": "9.4M",
        "mAP50": 0.8239,
        "mAP50_95": 0.4149,
        "fps": None,
    },
    "yolo11n_clahe": {
        "path": str(MODELS_DIR / "exp_yolo11n_clahe/train/weights/best.pt"),
        "display_name": "YOLO11n (CLAHE增强)",
        "params": "2.6M",
        "mAP50": 0.8071,
        "mAP50_95": 0.4034,
        "fps": None,
    },
    "yolo11s_clahe": {
        "path": str(MODELS_DIR / "exp_yolo11s_clahe/train/weights/best.pt"),
        "display_name": "YOLO11s (CLAHE增强)",
        "params": "9.4M",
        "mAP50": 0.8205,
        "mAP50_95": 0.4118,
        "fps": None,
    },
}

_model_cache: dict[str, YOLO] = {}


def get_model(model_name: str) -> YOLO:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS)}")
    if model_name not in _model_cache:
        _model_cache[model_name] = YOLO(AVAILABLE_MODELS[model_name]["path"])
    return _model_cache[model_name]


def detect(image: np.ndarray, model_name: str) -> tuple[list[dict], float, np.ndarray]:
    """Run detection. Returns ([detections], inference_time_ms, annotated_image)."""
    model = get_model(model_name)
    results = model(image, verbose=False)
    r = results[0]
    detections, inference_time = _parse_results(r, model)
    annotated = r.plot()
    return detections, inference_time, annotated


def _parse_results(r, model: YOLO) -> tuple[list[dict], float]:
    detections = []
    if r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()
        for i in range(len(boxes)):
            detections.append(
                {
                    "class_name": model.names[int(clss[i])],
                    "confidence": round(float(confs[i]), 4),
                    "x1": round(float(boxes[i][0]), 2),
                    "y1": round(float(boxes[i][1]), 2),
                    "x2": round(float(boxes[i][2]), 2),
                    "y2": round(float(boxes[i][3]), 2),
                }
            )
    inference_time = round(r.speed.get("inference", 0), 1)
    return detections, inference_time


# In-memory progress store for video processing
video_progress: dict[int, dict] = {}


def process_video(video_path: str, output_path: str, model_name: str, use_clahe: bool, task_id: int):
    """Process video frame by frame with YOLO detection. Updates video_progress dict."""
    from app.services.clahe import apply_clahe

    model = get_model(model_name)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    video_progress[task_id] = {"current": 0, "total": total_frames, "status": "processing"}
    total_detections = 0
    total_inference = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if use_clahe:
            frame = apply_clahe(frame)

        results = model(frame, verbose=False)
        r = results[0]
        dets, inf_time = _parse_results(r, model)
        total_detections += len(dets)
        total_inference += inf_time
        frame_count += 1

        annotated = r.plot()
        out.write(annotated)

        video_progress[task_id]["current"] = frame_count
        video_progress[task_id]["avg_inference_ms"] = round(total_inference / frame_count, 1)

    cap.release()
    out.release()

    video_progress[task_id]["status"] = "done"
    video_progress[task_id]["total_detections"] = total_detections
    video_progress[task_id]["avg_inference_ms"] = round(total_inference / max(frame_count, 1), 1)
