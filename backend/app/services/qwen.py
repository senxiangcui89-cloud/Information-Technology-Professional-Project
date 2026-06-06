import base64
import os
import httpx
from pathlib import Path


QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
QWEN_MODEL = "qwen-vl-max"


def analyze_detection(image_path: str, detections: list[dict], api_key: str | None = None) -> str:
    """Call Qwen-VL to analyze detection results and suggest cleanup measures."""
    key = api_key or QWEN_API_KEY
    if not key:
        return ""

    # Read and encode image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    det_summary = "\n".join(
        f"- {d['class_name']}: confidence={d['confidence']:.2%}, "
        f"bbox=({d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f})"
        for d in detections
    ) if detections else "No objects detected"

    prompt = f"""You are a water environment monitoring expert. The following are YOLO detection results for a water surface floating debris image:

Number of detected objects: {len(detections)}
Detection details:
{det_summary}

Based on the above detection results:
1. Analyze the current pollution status of the water area
2. Assess the pollution severity (Mild/Moderate/Severe)
3. Provide specific cleanup and remediation recommendations
4. If necessary, suggest follow-up monitoring measures

Please respond in English, within 200 words, with clear structure."""

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                QWEN_API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": QWEN_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            return ""
    except Exception:
        return ""
