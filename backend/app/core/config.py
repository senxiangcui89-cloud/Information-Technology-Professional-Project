from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SECRET_KEY = "floating-debris-detection-secret-key-2024"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440
DATABASE_URL = f"sqlite:///{PROJECT_ROOT / 'backend' / 'app.db'}"
UPLOAD_DIR = PROJECT_ROOT / "backend" / "uploads"
MODELS_DIR = PROJECT_ROOT / "experiments"
