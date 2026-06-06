from __future__ import annotations

import base64

import httpx

QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_API_KEY = "sk-396451b2f76f4ddfa1bbfb597445b8f5"  # Set via environment or config
QWEN_MODEL = "qwen-vl-max"


def analyze_detection(image_path: str, detections: list[dict], api_key: str | None = None) -> str:
    """Call Qwen-VL to analyze detection results and suggest cleanup measures."""
    key = api_key or QWEN_API_KEY
    if not key:
        return ""

    # Read and encode image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    det_summary = (
        "\n".join(
            f"- {d['class_name']}: confidence={d['confidence']:.2%}, "
            f"bbox=({d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f})"
            for d in detections
        )
        if detections
        else "未检测到任何目标"
    )

    prompt = f"""你是一个水环境监测专家。以下是对水面漂浮物图像进行YOLO检测的结果：

检测到的目标数量: {len(detections)}
检测详情:
{det_summary}

请根据以上检测结果：
1. 分析当前水域的漂浮物污染状况
2. 评估污染严重程度（轻/中/重）
3. 提供具体的清理和治理建议
4. 如有必要，建议后续监测措施

请用中文回答，控制在200字以内，条理清晰。"""

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
