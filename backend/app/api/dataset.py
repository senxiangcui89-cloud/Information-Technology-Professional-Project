from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_current_user
from app.core.config import UPLOAD_DIR
from app.services.dataset import extract_and_validate, list_datasets

router = APIRouter(prefix="/api/dataset", tags=["dataset"])

DATASET_DIR = UPLOAD_DIR / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = None,
    user: dict = Depends(get_current_user),
):
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(400, "请上传 ZIP 格式文件")

    dataset_name = name or Path(file.filename).stem
    safe_name = dataset_name.replace(" ", "_").replace("/", "_")
    zip_path = DATASET_DIR / f"{uuid.uuid4().hex}.zip"

    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        info = extract_and_validate(str(zip_path), safe_name)
        zip_path.unlink()
        return {"name": safe_name, **info}
    except Exception as e:
        zip_path.unlink(missing_ok=True)
        raise HTTPException(400, f"数据集解析失败: {e!s}")


@router.get("/list")
def list_all(user: dict = Depends(get_current_user)):
    return list_datasets()


@router.delete("/{name}")
def delete_dataset(name: str, user: dict = Depends(get_current_user)):
    target = DATASET_DIR / name
    if not target.exists():
        raise HTTPException(404, "数据集不存在")
    shutil.rmtree(target)
    return {"message": f"数据集 {name} 已删除"}
