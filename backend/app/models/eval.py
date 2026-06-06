from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class EvalTask(Base):
    __tablename__ = "eval_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_name = Column(String(50), nullable=False)
    dataset_name = Column(String(100), nullable=False)
    image_count = Column(Integer, nullable=False)
    status = Column(String(20), default="pending")
    # metrics JSON: {mAP50, mAP50-95, precision, recall, fps, per_class: {...}}
    metrics = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
