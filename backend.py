import os
from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
from ultralytics import YOLO

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 🔥 Load custom fire model if it exists, otherwise use base YOLOv8s
MODEL_PATH = "runs/detect/fire_detection_v8/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8s.pt" # Fallback for now

model = YOLO(MODEL_PATH)
print(f"✅ Loaded model from: {MODEL_PATH}")

# ── STATUS API ──
@app.get("/status")
def get_status():
    return {
        "status": "online",
        "model": MODEL_PATH
    }


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            data = await websocket.receive_bytes()

            # convert image
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Run YOLO
            results = model(frame, conf=0.4) # filter by confidence

            detections = []
            fire_detected = False
            smoke_detected = False
            max_conf = 0.0

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]
                    
                    # YOLOv8 custom classes usually: 0=fire, 1=smoke
                    # Keep track if they appear
                    if label.lower() == "fire": fire_detected = True
                    if label.lower() == "smoke": smoke_detected = True
                    if conf > max_conf: max_conf = conf

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "class": label,
                        "conf": round(conf, 2),
                        "box": [x1, y1, x2, y2]
                    })

            await websocket.send_json({
                "fire_detected": fire_detected,
                "smoke_detected": smoke_detected,
                "confidence": round(max_conf, 2),
                "detections": detections
            })
        except Exception as e:
            print(f"Error: {e}")
            break