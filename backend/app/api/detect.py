import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import UPLOAD_DIR
from app.api.deps import get_current_user
from app.models.detection import DetectionTask, DetectionResult
from app.schemas.detect import (
    DetectResponse, DetectionBox, ModelInfo,
    VideoProgress, AnalyzeResponse, ReportResponse,
)
from app.services.detector import detect, process_video, video_progress, AVAILABLE_MODELS
from app.services.clahe import apply_clahe
from app.services.qwen import analyze_detection
from app.services.report import generate_report

router = APIRouter(prefix="/api/detect", tags=["detect"])

RESULT_DIR = UPLOAD_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
(UPLOAD_DIR / "reports").mkdir(parents=True, exist_ok=True)
VIDEO_DIR = UPLOAD_DIR / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/models", response_model=list[ModelInfo])
def list_models():
    return [
        ModelInfo(
            name=key,
            display_name=v["display_name"],
            params=v["params"],
            mAP50=v["mAP50"],
            mAP50_95=v["mAP50_95"],
            fps=v["fps"],
        )
        for key, v in AVAILABLE_MODELS.items()
    ]


@router.post("/image", response_model=DetectResponse)
def detect_image(
    file: UploadFile = File(...),
    model_name: str = Form("yolo11n_raw"),
    use_clahe: bool = Form(False),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model: {model_name}")

    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "无法解码图片")

    if use_clahe:
        image = apply_clahe(image)

    detections, inference_time, annotated = detect(image, model_name)

    result_filename = f"{uuid.uuid4().hex}.jpg"
    result_path = RESULT_DIR / result_filename
    cv2.imwrite(str(result_path), annotated)

    task = DetectionTask(
        user_id=int(user["sub"]),
        model_name=model_name,
        source_type="image",
        source_filename=file.filename or "upload.jpg",
        use_clahe=use_clahe,
        status="done",
        result_filename=result_filename,
        detection_count=len(detections),
        inference_time_ms=inference_time,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    for d in detections:
        db.add(DetectionResult(
            task_id=task.id,
            class_id=0,
            class_name=d["class_name"],
            confidence=d["confidence"],
            x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
        ))
    db.commit()

    return DetectResponse(
        task_id=task.id,
        status="done",
        model_name=model_name,
        detection_count=len(detections),
        inference_time_ms=inference_time,
        detections=[DetectionBox(**d) for d in detections],
        result_image_url=f"/uploads/results/{result_filename}",
    )


@router.post("/video", response_model=DetectResponse)
def detect_video(
    file: UploadFile = File(...),
    model_name: str = Form("yolo11n_raw"),
    use_clahe: bool = Form(False),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model: {model_name}")

    # Save uploaded video
    video_filename = f"{uuid.uuid4().hex}_{file.filename or 'video.mp4'}"
    video_path = VIDEO_DIR / video_filename
    with open(video_path, "wb") as f:
        f.write(file.file.read())

    # Verify video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(400, "无法读取视频文件")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Create task
    output_filename = f"det_{video_filename}"
    output_path = str(VIDEO_DIR / output_filename)

    task = DetectionTask(
        user_id=int(user["sub"]),
        model_name=model_name,
        source_type="video",
        source_filename=file.filename or "video.mp4",
        use_clahe=use_clahe,
        status="processing",
        result_filename=output_filename,
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    # Process in background thread
    thread = threading.Thread(
        target=_process_video_task,
        args=(str(video_path), output_path, model_name, use_clahe, task.id),
        daemon=True,
    )
    thread.start()

    return DetectResponse(
        task_id=task.id,
        status="processing",
        model_name=model_name,
        detection_count=0,
        inference_time_ms=0,
        detections=[],
        result_video_url=None,
        message=f"视频共 {total_frames} 帧，正在处理中...",
    )


def _process_video_task(video_path: str, output_path: str, model_name: str, use_clahe: bool, task_id: int):
    """Background video processing, updates DB on completion."""
    from app.core.database import SessionLocal
    process_video(video_path, output_path, model_name, use_clahe, task_id)

    db = SessionLocal()
    try:
        task = db.query(DetectionTask).get(task_id)
        if task:
            prog = video_progress.get(task_id, {})
            task.status = "done"
            task.detection_count = prog.get("total_detections", 0)
            task.inference_time_ms = prog.get("avg_inference_ms", 0)
            db.commit()
    finally:
        db.close()


@router.get("/progress/{task_id}", response_model=VideoProgress)
def get_progress(task_id: int):
    prog = video_progress.get(task_id)
    if not prog:
        raise HTTPException(404, "任务不存在或已完成")
    return VideoProgress(
        task_id=task_id,
        status=prog["status"],
        current_frame=prog["current"],
        total_frames=prog["total"],
        avg_inference_ms=prog.get("avg_inference_ms"),
        total_detections=prog.get("total_detections"),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    task_id: int = Form(...),
    api_key: str | None = Form(None),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = db.query(DetectionTask).get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    results = db.query(DetectionResult).filter(DetectionResult.task_id == task_id).all()
    detections = [
        {"class_name": r.class_name, "confidence": r.confidence,
         "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2}
        for r in results
    ]

    image_path = RESULT_DIR / task.result_filename if task.result_filename else None
    if not image_path or not image_path.exists():
        # Fallback: look for source file
        image_path = UPLOAD_DIR / "results" / (task.result_filename or "")

    analysis = ""
    if image_path and image_path.exists():
        analysis = analyze_detection(str(image_path), detections, api_key or None)

    return AnalyzeResponse(analysis=analysis or "AI 分析暂不可用，请配置 Qwen API Key。")


@router.get("/export-report/{task_id}", response_model=ReportResponse)
def export_report(
    task_id: int,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = db.query(DetectionTask).get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    results = db.query(DetectionResult).filter(DetectionResult.task_id == task_id).all()
    detections = [
        {"class_name": r.class_name, "confidence": r.confidence,
         "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2}
        for r in results
    ]

    result_image_path = RESULT_DIR / task.result_filename if task.result_filename else ""
    report_path = generate_report(
        image_path="",
        result_image_path=str(result_image_path) if result_image_path else "",
        detections=detections,
        model_name=task.model_name,
        inference_time_ms=task.inference_time_ms or 0,
        use_clahe=task.use_clahe,
    )

    report_name = Path(report_path).name
    return ReportResponse(report_url=f"/uploads/reports/{report_name}")


@router.get("/result/{task_id}", response_model=DetectResponse)
def get_result(task_id: int, db: Session = Depends(get_db)):
    task = db.query(DetectionTask).get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    results = db.query(DetectionResult).filter(DetectionResult.task_id == task_id).all()
    url_field = None
    if task.result_filename:
        if task.source_type == "video":
            url_field = f"/uploads/videos/{task.result_filename}"
        else:
            url_field = f"/uploads/results/{task.result_filename}"

    return DetectResponse(
        task_id=task.id,
        status=task.status,
        model_name=task.model_name,
        detection_count=task.detection_count or 0,
        inference_time_ms=task.inference_time_ms or 0,
        detections=[
            DetectionBox(
                class_name=r.class_name,
                confidence=r.confidence,
                x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2,
            ) for r in results
        ],
        result_image_url=url_field if task.source_type != "video" else None,
        result_video_url=url_field if task.source_type == "video" else None,
    )


@router.get("/tasks")
def list_tasks(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    tasks = (
        db.query(DetectionTask)
        .filter(DetectionTask.user_id == int(user["sub"]))
        .order_by(DetectionTask.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "task_id": t.id,
            "model_name": t.model_name,
            "source_type": t.source_type,
            "source_filename": t.source_filename,
            "status": t.status,
            "detection_count": t.detection_count,
            "inference_time_ms": t.inference_time_ms,
            "result_video_url": f"/uploads/videos/{t.result_filename}" if t.source_type == "video" and t.result_filename else None,
            "result_image_url": f"/uploads/results/{t.result_filename}" if t.source_type == "image" and t.result_filename else None,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in tasks
    ]
