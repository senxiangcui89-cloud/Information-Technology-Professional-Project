import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine, Base
from app.api.auth import router as auth_router
from app.api.detect import router as detect_router
from app.api.dataset import router as dataset_router
from app.api.eval import router as eval_router
from app.api.camera import router as camera_router
from app.api.admin import router as admin_router
from app.core.config import UPLOAD_DIR
from app.models.eval import EvalTask  # ensure table created

Base.metadata.create_all(bind=engine)

app = FastAPI(title="水面漂浮物检测系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(detect_router)
app.include_router(dataset_router)
app.include_router(eval_router)
app.include_router(camera_router)
app.include_router(admin_router)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
