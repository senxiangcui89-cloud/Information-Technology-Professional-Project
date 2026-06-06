from pydantic import BaseModel


class DetectionBox(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class DetectResponse(BaseModel):
    task_id: int
    status: str
    model_name: str
    detection_count: int
    inference_time_ms: float
    detections: list[DetectionBox]
    result_image_url: str | None = None
    result_video_url: str | None = None
    message: str | None = None


class ModelInfo(BaseModel):
    name: str
    display_name: str
    params: str
    mAP50: float | None = None
    mAP50_95: float | None = None
    fps: float | None = None


class VideoProgress(BaseModel):
    task_id: int
    status: str
    current_frame: int
    total_frames: int
    avg_inference_ms: float | None = None
    total_detections: int | None = None


class AnalyzeResponse(BaseModel):
    analysis: str


class ReportResponse(BaseModel):
    report_url: str
