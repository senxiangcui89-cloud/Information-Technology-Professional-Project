# Water Surface Floating Debris Detection System

Deep learning-based water surface floating debris detection system using YOLO, supporting image/video detection, real-time camera monitoring, CLAHE image enhancement, AI-powered analysis, and report export.

## Features

- **Image Detection** — Upload an image to automatically detect and annotate floating debris
- **Video Detection** — Upload a video for frame-by-frame detection with real-time progress tracking
- **Real-time Camera** — Browser camera streaming with frame-by-frame detection
- **CLAHE Enhancement** — Optional image preprocessing for improved detection in low-contrast scenes
- **Multi-Model** — YOLOv8 / YOLO11, raw data / CLAHE-enhanced, switchable models
- **AI Analysis** — Integrated Qwen-VL (Tongyi Qianwen) for pollution assessment and cleanup recommendations
- **Model Evaluation** — mAP, Precision, Recall, F1-Score metrics
- **Dataset Management** — Upload, annotate, and split datasets
- **Report Export** — Auto-generate Word detection reports
- **User System** — Registration / Login / JWT authentication with admin panel

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI (Python) |
| Database | SQLite + SQLAlchemy ORM |
| Object Detection | Ultralytics YOLO (v8/v11) |
| Image Processing | OpenCV, CLAHE |
| AI Analysis | Alibaba Cloud Qwen-VL |
| Frontend | Vue 3 + TypeScript |
| UI Library | Element Plus |
| Charts | ECharts |
| Build Tool | Vite |
| Model Training | PyTorch, Ultralytics YOLO |

## Project Structure

```
├── backend/                  # FastAPI backend
│   ├── main.py               # App entry point
│   └── app/
│       ├── api/               # API routes
│       │   ├── detect.py      # Image & video detection
│       │   ├── camera.py      # Real-time camera detection
│       │   ├── dataset.py     # Dataset management
│       │   ├── eval.py        # Model evaluation
│       │   ├── auth.py        # User authentication
│       │   └── admin.py       # Admin panel
│       ├── core/              # Config & database
│       ├── models/            # ORM data models
│       ├── schemas/           # Pydantic request/response models
│       └── services/          # Business logic
│           ├── detector.py    # YOLO detection engine
│           ├── clahe.py       # CLAHE image enhancement
│           ├── qwen.py        # Qwen-VL AI analysis
│           ├── report.py      # Report generation
│           └── evaluator.py   # Model evaluation
├── frontend/                  # Vue 3 frontend
│   └── src/views/
│       ├── Detect.vue         # Image detection page
│       ├── VideoDetect.vue    # Video detection page
│       ├── Camera.vue         # Real-time camera detection
│       ├── DatasetManager.vue # Dataset management
│       ├── ModelEval.vue      # Model evaluation
│       ├── History.vue        # Detection history
│       ├── Admin.vue          # Admin panel
│       ├── Login.vue          # Login page
│       └── Register.vue       # Registration page
├── project/                   # Training & inference tools
│   ├── config/                # Training config files
│   ├── data/                  # Data preprocessing utilities
│   ├── train/                 # Training pipeline (batch experiments)
│   ├── inference/             # Image/video inference scripts
│   ├── models/                # Detector wrapper & model export
│   ├── eval/                  # Model evaluation scripts
│   └── utils/                 # Logging & visualization utilities
└── requirements.txt           # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/macOS

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Prepare Model Weights

Place trained YOLO model weights (`.pt` files) under the `experiments/` directory:

```
experiments/
├── exp_yolo11n_raw/train/weights/best.pt
├── exp_yolo11s_raw/train/weights/best.pt
├── exp_yolo11n_clahe/train/weights/best.pt
└── exp_yolo11s_clahe/train/weights/best.pt
```

### 3. Start Backend

```bash
cd backend
python main.py
# Server runs at http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
# Frontend runs at http://localhost:5173
```

### 5. Model Training (Optional)

```bash
# Single training run
python -m project.train.train

# Batch comparison experiments (raw vs CLAHE, multiple models)
python -m project.train.train --run-all

# Image inference
python -m project.inference.detect_image --weights experiments/exp_yolo11n_raw/best.pt --source test.jpg

# Video inference
python -m project.inference.detect_video --weights experiments/exp_yolo11n_raw/best.pt --source test.mp4
```

## API Overview

| Endpoint | Description |
|----------|-------------|
| `POST /api/auth/register` | User registration |
| `POST /api/auth/login` | User login |
| `POST /api/detect/image` | Image detection |
| `POST /api/detect/video` | Video detection |
| `GET /api/detect/progress/{id}` | Video processing progress |
| `POST /api/detect/analyze` | AI analysis of detection results |
| `GET /api/detect/export-report/{id}` | Export detection report |
| `POST /api/camera/frame` | Camera frame detection |
| `GET /api/datasets` | List datasets |
| `GET /api/eval/models` | Model evaluation metrics |
| `GET /api/health` | Health check |

## Dataset

This project uses the **FloW-Dataset** for water surface floating debris, covering the following categories:

- Water grass
- Floating debris
- Duckweed
- Oil slick

## Configuration

Edit `project/config/default.yaml` to adjust:

- Dataset paths and split ratios
- CLAHE preprocessing parameters
- Model candidates
- Training hyperparameters (epochs, batch_size, learning rate, optimizer, etc.)
- Evaluation metrics and thresholds

## License

MIT
