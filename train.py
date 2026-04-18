from ultralytics import YOLO

def train_fire_model():
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Train the model using the custom dataset
    # We use data.yaml which defines 'fire' and 'smoke' classes
    results = model.train(
        data="datasets/data.yaml", 
        epochs=50, 
        imgsz=640, 
        batch=16,
        name="fire_detection_v8"
    )
    
    print("✅ Training complete. The best weights are saved in runs/detect/fire_detection_v8/weights/best.pt")

if __name__ == "__main__":
    train_fire_model()
