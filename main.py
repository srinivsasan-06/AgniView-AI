import os
from fastapi import FastAPI, WebSocket
import cv2
from ultralytics import YOLO
import base64
import numpy as np

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 🔥 Load custom fire model if it exists
MODEL_PATH = "runs/detect/fire_detection_v8/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    while True:
        try:
            data = await ws.receive_text()

            # Convert base64 → image
            img_data = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Run YOLO
            results = model(frame, conf=0.4)

            fire_detected = False
            smoke_detected = False
            confidence = 0.0
            detections = []

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

            await ws.send_json({
                "fire_detected": fire_detected,
                "smoke_detected": smoke_detected,
                "confidence": round(confidence, 2),
                "detections": detections
            })
        except Exception as e:
            print(f"Error: {e}")
            break