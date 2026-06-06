"""Inference prototype: detect bottles in images/video and save results."""
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_image(model, source, output_dir):
    results = model(source)
    for r in results:
        out_path = output_dir / Path(r.path).name
        cv2.imwrite(str(out_path), r.plot())
        print(f"Saved: {out_path}")

def run_video(model, source, output_dir):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        str(output_dir / f"det_{Path(source).name}"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        out.write(results[0].plot())
    cap.release()
    out.release()
    print(f"Saved: {output_dir / f'det_{Path(source).name}'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--source", required=True, help="Image/video file or directory")
    parser.add_argument("--output", default="experiments/inference_output", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    ext = Path(args.source).suffix.lower()

    if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
        run_image(model, args.source, output_dir)
    elif ext in {".mp4", ".avi", ".mov", ".mkv"}:
        run_video(model, args.source, output_dir)
    else:
        # Directory mode
        for img_path in Path(args.source).glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                run_image(model, str(img_path), output_dir)
