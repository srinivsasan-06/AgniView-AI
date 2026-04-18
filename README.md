# AgniView-AI

An intelligent, real-time fire and smoke detection system utilizing YOLOv8 and FastAPI to provide automated surveillance and instant alerting for forest conservation and disaster prevention.

## About the Project

AgniView AI is a real-time computer vision system that detects early signs of fire and smoke from live camera feeds.

Traditional monitoring systems such as watchtowers and satellite-based observation often introduce delays, allowing small incidents to become major disasters.

AgniView AI solves this problem by using deep learning and real-time video analysis to immediately identify threats and trigger alerts.

---

## Features

* Real-time fire and smoke detection
* YOLOv8-based object detection
* Live camera stream analysis
* Bounding boxes with confidence scores
* Instant alert generation
* Dark-mode monitoring dashboard
* Real-time analytics and event tracking

---

## Tech Stack

```md
AI / ML      : CNN, YOLOv8
Backend      : FastAPI
Frontend     : React, Tailwind CSS
Streaming    : MJPEG
```

---

## Workflow

```text
Capture  -> Process  -> Alert  -> Visualize
```

### 1. Capture

The system receives live video input from cameras.

### 2. Process

Each frame is analyzed using YOLOv8.

* Fire and smoke regions are detected
* Bounding boxes are generated
* Confidence scores are calculated

### 3. Alert

If fire or smoke is detected above a threshold, the backend sends an alert immediately.

### 4. Visualize

The React dashboard displays:

* Live video stream
* Detection overlays
* Confidence levels
* Analytics and alert history

---

## Repository Structure

```text
AgniView-AI/
│
├── backend/
│   ├── main.py
│   ├── detection/
│   ├── models/
│   └── alerts/
│
├── frontend/
│   ├── src/
│   ├── components/
│   └── pages/
│
├── assets/
├── README.md
└── requirements.txt
```

---

## Team

* Srinivasan S
* Gayathri V
* Yugesh Kumar M

---

## Repository Topics

```text
computer-vision
yolov8
fire-detection
fastapi
react
ai-for-good
deep-learning
real-time-analytics
```

---

## Future Enhancements

* SMS / Email alert integration
* Multi-camera support
* Forest fire heatmap visualization
* Mobile app support
* Cloud deployment

---

## License

This project is developed for educational and research purposes.
