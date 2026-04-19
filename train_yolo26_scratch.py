from ultralytics import YOLO
import os

def train_from_scratch():
    # PATHS
    MODEL_PATH = "/home/students/bigdrive/Drone/T02_HIT_UAV_Det/yolo26m.pt"
    DATA_PATH = "/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/data/auair_norm.yaml"
    EXPORT_PATH = "/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train"
    
    # Initialize model
    model = YOLO(MODEL_PATH)

    # OFFICIAL MASTER RUN (800px, Normalized Dataset, 100 Epochs)
    results = model.train(
        data=DATA_PATH,
        epochs=100,
        imgsz=800,  # Starting with high-res from Epoch 1
        batch=8,    # Stable batch for 800px on L40S
        lr0=0.01,   # Standard Starting Learning Rate
        project=EXPORT_PATH,
        name="yolo26m_auair_final_scratch",
        device=0,
        mosaic=1.0,
        mixup=0.1,
        cos_lr=True,
        close_mosaic=10,
        patience=50
    )

if __name__ == "__main__":
    train_from_scratch()
