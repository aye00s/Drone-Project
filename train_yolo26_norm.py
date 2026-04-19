from ultralytics import YOLO

def train_normalized():
    # Load the custom YOLOv26m model
    model_path = "/home/students/bigdrive/Drone/T02_HIT_UAV_Det/yolo26m.pt"
    model = YOLO(model_path)

    # Train the model
    results = model.train(
        data="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/data/auair_norm.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        project="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train",
        name="yolo26m_auair_norm",
        device=0,
        mosaic=1.0,  # Maintain high augmentation
        mixup=0.1
    )

if __name__ == "__main__":
    train_normalized()
