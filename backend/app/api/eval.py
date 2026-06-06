import threading
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Form
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import UPLOAD_DIR
from app.api.deps import get_current_user
from app.models.eval import EvalTask
from app.services.evaluator import run_evaluation
from app.services.dataset import list_datasets
from app.services.detector import AVAILABLE_MODELS

router = APIRouter(prefix="/api/eval", tags=["eval"])

EVAL_RESULTS_DIR = UPLOAD_DIR / "eval_results"
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/run")
def eval_run(
    model_name: str = Form(...),
    dataset_name: str = Form(...),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(400, f"Unknown model: {model_name}")

    datasets = list_datasets()
    ds = next((d for d in datasets if d["name"] == dataset_name), None)
    if not ds:
        raise HTTPException(400, f"Dataset not found: {dataset_name}")
    if not ds.get("valid"):
        raise HTTPException(400, "Invalid dataset structure: requires images/train and images/val directories")

    task = EvalTask(
        user_id=int(user["sub"]),
        model_name=model_name,
        dataset_name=dataset_name,
        image_count=ds["val_images"],
        status="processing",
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    def _run():
        from app.core.database import SessionLocal
        db2 = SessionLocal()
        try:
            dataset_path = UPLOAD_DIR / "datasets" / dataset_name
            metrics = run_evaluation(model_name, str(dataset_path), task.id)
            t = db2.query(EvalTask).get(task.id)
            if t:
                t.metrics = metrics
                t.status = "done"
                db2.commit()
        except Exception as e:
            t = db2.query(EvalTask).get(task.id)
            if t:
                t.status = "failed"
                t.error_message = str(e)
                db2.commit()
        finally:
            db2.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "task_id": task.id,
        "status": "processing",
        "model_name": model_name,
        "dataset_name": dataset_name,
        "image_count": ds["val_images"],
    }


@router.get("/result/{task_id}")
def eval_result(task_id: int, user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(EvalTask).get(task_id)
    if not task:
        raise HTTPException(404, "Evaluation task not found")
    return {
        "task_id": task.id,
        "status": task.status,
        "model_name": task.model_name,
        "dataset_name": task.dataset_name,
        "image_count": task.image_count,
        "metrics": task.metrics,
        "error_message": task.error_message,
        "created_at": task.created_at.isoformat() if task.created_at else None,
    }


@router.get("/tasks")
def list_eval_tasks(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    tasks = (
        db.query(EvalTask)
        .order_by(EvalTask.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "task_id": t.id,
            "status": t.status,
            "model_name": t.model_name,
            "dataset_name": t.dataset_name,
            "image_count": t.image_count,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in tasks
    ]


@router.get("/options")
def eval_options():
    return {
        "models": [
            {"name": k, "display_name": v["display_name"], "mAP50": v["mAP50"]}
            for k, v in AVAILABLE_MODELS.items()
        ],
        "datasets": list_datasets(),
    }
