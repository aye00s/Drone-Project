from ultralytics import YOLO
import os

def boost_training():
    # Path to the last checkpoint of the normalized run
    checkpoint_path = "/mnt/8c85412b-1aea-4453-b8b1-d576629fabc0/Drone/T03_AU_AIR_Traffic/runs/train/yolo26m_auair_norm/weights/last.pt"
    
    if not os.path.exists(checkpoint_path):
        print("Error: Checkpoint not found.")
        return

    # Load the model from checkpoint
    model = YOLO(checkpoint_path)

    # Resume training with BOOSTED parameters
    # 1. Higher resolution (800)
    # 2. Multi-scale enabled
    # 3. Aggressive augmentation
    # 4. Learning rate restart (lr0=0.001)
    results = model.train(
        data="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/data/auair_norm.yaml",
        epochs=100, # Total epochs to reach
        imgsz=800,  # BOOSTED from 640
        batch=8,    # Lowered batch size to accommodate higher imgsz
        lr0=0.001,  # Fine-tuning starting point
        project="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train",
        name="yolo26m_auair_norm_boost_v2", # New run name
        device=0,
        mosaic=1.0,
        mixup=0.2,
        cos_lr=True,
        close_mosaic=10
    )

if __name__ == "__main__":
    boost_training()
