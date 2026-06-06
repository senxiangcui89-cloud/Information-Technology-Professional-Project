from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from app.core.database import Base


class DetectionTask(Base):
    __tablename__ = "detection_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_name = Column(String(50), nullable=False)
    source_type = Column(String(20), nullable=False)  # image / video / camera
    source_filename = Column(String(255), nullable=False)
    use_clahe = Column(Boolean, default=False)
    status = Column(String(20), default="pending")  # pending / processing / done / failed
    result_filename = Column(String(255), nullable=True)
    detection_count = Column(Integer, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DetectionResult(Base):
    __tablename__ = "detection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Integer, ForeignKey("detection_tasks.id"), nullable=False)
    class_id = Column(Integer, nullable=False)
    class_name = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float, nullable=False)
    y1 = Column(Float, nullable=False)
    x2 = Column(Float, nullable=False)
    y2 = Column(Float, nullable=False)
