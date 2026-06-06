from docx import Document
from docx.shared import Inches
from pathlib import Path
from datetime import datetime
from app.core.config import UPLOAD_DIR


def generate_report(
    image_path: str,
    result_image_path: str,
    detections: list[dict],
    model_name: str,
    inference_time_ms: float,
    use_clahe: bool,
    ai_analysis: str | None = None,
) -> str:
    doc = Document()
    doc.add_heading("Floating Debris Detection Report", level=1)
    doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Model: {model_name}")
    doc.add_paragraph(f"CLAHE Preprocessing: {'Enabled' if use_clahe else 'Disabled'}")
    doc.add_paragraph(f"Inference Time: {inference_time_ms:.1f} ms")
    doc.add_paragraph(f"Detection Count: {len(detections)}")

    doc.add_heading("Detection Details", level=2)
    if detections:
        table = doc.add_table(rows=1, cols=4)
        hdr = table.rows[0].cells
        hdr[0].text = "Class"
        hdr[1].text = "Confidence"
        hdr[2].text = "Bounding Box (x1,y1,x2,y2)"
        hdr[3].text = "Area"
        for d in detections:
            row = table.add_row().cells
            row[0].text = d["class_name"]
            row[1].text = f"{d['confidence']:.2%}"
            row[2].text = f"({d['x1']:.0f},{d['y1']:.0f},{d['x2']:.0f},{d['y2']:.0f})"
            area = (d["x2"] - d["x1"]) * (d["y2"] - d["y1"])
            row[3].text = f"{area:.0f} px²"
    else:
        doc.add_paragraph("No objects detected.")

    if ai_analysis:
        doc.add_heading("AI Analysis & Recommendations", level=2)
        doc.add_paragraph(ai_analysis)

    if Path(result_image_path).exists():
        doc.add_heading("Detection Result Image", level=2)
        doc.add_picture(result_image_path, width=Inches(5))

    output_dir = UPLOAD_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    output_path = output_dir / filename
    doc.save(str(output_path))
    return str(output_path)
