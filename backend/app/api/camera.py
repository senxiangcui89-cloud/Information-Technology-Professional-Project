import base64
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import JSONResponse

from app.api.deps import get_current_user
from app.services.detector import detect, AVAILABLE_MODELS

router = APIRouter(prefix="/api/camera", tags=["camera"])


@router.post("/frame")
def detect_frame(
    image_b64: str = Form(...),
    model_name: str = Form("yolo11n_raw"),
    user: dict = Depends(get_current_user),
):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model: {model_name}")

    try:
        # Decode base64 JPEG
        img_data = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "无法解码图像帧")
    except Exception:
        raise HTTPException(400, "Base64 解码失败")

    detections, inference_time, annotated = detect(frame, model_name)

    # Encode annotated frame back to base64
    _, buffer = cv2.imencode(".jpg", annotated)
    result_b64 = base64.b64encode(buffer).decode()

    return {
        "detections": [
            {"class_name": d["class_name"], "confidence": d["confidence"],
             "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"]}
            for d in detections
        ],
        "count": len(detections),
        "inference_ms": inference_time,
        "image_b64": result_b64,
    }
