# ── IMPORTS ──
from fastapi import FastAPI
from ultralytics import YOLO
import os
import numpy as np
import cv2

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── MODEL LOAD ──
MODEL_PATH = "runs/detect/fire_detection_v8/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt" # Fallback

model = YOLO(MODEL_PATH)
print(f"🔥 Model loaded from: {MODEL_PATH}")


# ── GLOBAL VARIABLES ──
fire_detected = False
smoke_detected = False
confidence = 0.0


# ── STATUS API ──
@app.get("/status")
def get_status():
    global fire_detected, smoke_detected, confidence
    return {
        "fire_detected": fire_detected,
        "smoke_detected": smoke_detected,
        "confidence": confidence
    }


# ── DETECTION API ──
@app.post("/detect")
async def detect(frame: bytes):
    global fire_detected, smoke_detected, confidence

    # Convert frame
    nparr = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run YOLO
    results = model(img, conf=0.4)

    fire_detected = False
    smoke_detected = False
    confidence = 0.0
    detections = []

    # 🔥 DETECTION LOGIC
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            if label.lower() == "fire": fire_detected = True
            if label.lower() == "smoke": smoke_detected = True
            if conf > confidence: confidence = conf

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class": label,
                "conf": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

    return {
        "fire_detected": fire_detected,
        "smoke_detected": smoke_detected,
        "confidence": round(confidence, 2),
        "detections": detections
    }