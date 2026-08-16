# app.py — Car Damage Analyzer (YOLOv8 + OpenAI LLM)
# Hugging Face Spaces compatible

import os
import time
from pathlib import Path
from typing import List, Dict

import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # allow app to run without OpenAI installed


# ==================================================
# Environment fixes for Hugging Face Spaces
# ==================================================
# Ultralytics needs a writable config directory
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"


# ==================================================
# Paths
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cardd_yolov8n_best.pt"

if not MODEL_PATH.exists():
    models_dir = BASE_DIR / "models"
    existing = [p.name for p in models_dir.glob("*")] if models_dir.exists() else []
    raise FileNotFoundError(
        f"Missing YOLO weights at {MODEL_PATH}. "
        f"models/ contains: {existing}"
    )


# ==================================================
# Load YOLO model
# ==================================================
yolo = YOLO(str(MODEL_PATH))


# ==================================================
# OpenAI client (optional)
# ==================================================
client = None
if OpenAI and os.getenv("OPENAI_API_KEY"):
    client = OpenAI()

_LAST_CALL_TS = 0.0
MIN_SECONDS_BETWEEN_LLM_CALLS = 3.0  # protect API credits


# ==================================================
# Rule-based report (fallback)
# ==================================================
def rule_based_report(result) -> str:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return "No visible damage detected in this image."

    counts: Dict[str, int] = {}
    for box in boxes:
        cls_id = int(box.cls)
        cls_name = yolo.names[cls_id]
        counts[cls_name] = counts.get(cls_name, 0) + 1

    total = sum(counts.values())
    severity = "High" if total >= 5 else "Moderate" if total >= 3 else "Mild"

    lines = ["Detected damage types:"]
    for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {v} × {k}")

    lines += ["", f"Overall severity: {severity}.", "", "Recommended action:"]

    if "glass shatter" in counts or "lamp broken" in counts:
        lines.append("- Replace damaged glass or lamps for safety and legal compliance.")
    if "dent" in counts or "crack" in counts:
        lines.append("- Body shop inspection recommended; panel repair may be needed.")
    if "scratch" in counts and total <= 2:
        lines.append("- Likely cosmetic; polishing or repainting may suffice.")

    if len(lines) <= 6:
        lines.append("- Manual inspection recommended for accurate repair estimation.")

    return "\n".join(lines)


# ==================================================
# Prepare detections for LLM
# ==================================================
def detections_for_llm(result) -> List[Dict]:
    dets = []
    boxes = result.boxes
    if boxes is None:
        return dets

    for box in boxes:
        cls_id = int(box.cls)
        dets.append(
            {
                "class_name": yolo.names[cls_id],
                "confidence": round(float(box.conf), 3),
            }
        )
    return dets


# ==================================================
# OpenAI LLM report
# ==================================================
def llm_report(dets: List[Dict], style: str) -> str | None:
    global _LAST_CALL_TS

    if client is None:
        return None

    now = time.time()
    if now - _LAST_CALL_TS < MIN_SECONDS_BETWEEN_LLM_CALLS:
        return "Please wait a few seconds before requesting another LLM report."
    _LAST_CALL_TS = now

    if not dets:
        return "No visible damage detected in this image."

    det_lines = "\n".join(
        f"- {d['class_name']} (confidence {d['confidence']})" for d in dets
    )

    role = "a professional vehicle workshop technician" if style == "workshop" else "an insurance claim assessor"

    prompt = f"""
You are {role}.

A YOLO-based car damage detector produced these detections:
{det_lines}

Write a short report including:
- Damage types present
- Severity (mild / moderate / severe)
- 2–3 clear recommendations
- If exact location on the car cannot be inferred, say so

Use bullet points. Be concise and professional.
""".strip()

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=220,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}"


# ==================================================
# Main inference function
# ==================================================
def analyze(image: np.ndarray, conf: float, use_llm: bool, style: str):
    if image is None:
        return None, "Please upload an image."

    pil = Image.fromarray(image.astype("uint8"), "RGB")
    results = yolo.predict(source=pil, imgsz=640, conf=float(conf), verbose=False)
    r = results[0]

    annotated = r.plot()  # BGR
    annotated_pil = Image.fromarray(annotated[..., ::-1])  # RGB

    if use_llm:
        dets = detections_for_llm(r)
        text = llm_report(dets, style)
        if text:
            return annotated_pil, text

    return annotated_pil, rule_based_report(r)


# ==================================================
# Gradio UI
# ==================================================
demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(type="numpy", label="Upload car image"),
        gr.Slider(0.05, 0.80, value=0.30, step=0.05, label="YOLO confidence threshold"),
        gr.Checkbox(value=True, label="Use LLM-generated report (requires API key)"),
        gr.Radio(
            choices=["workshop", "insurance"],
            value="workshop",
            label="Report style",
        ),
    ],
    outputs=[
        gr.Image(type="pil", label="Detected damage"),
        gr.Textbox(lines=12, label="Damage report"),
    ],
    title="Car Damage Analyzer (YOLOv8 + LLM)",
    description="Public demo: YOLO detects vehicle damage; optional LLM generates a human-readable report.",
)


if __name__ == "__main__":
    demo.launch()
