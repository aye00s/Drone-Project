from ultralytics import YOLO
import os

def precision_boost():
    # PATHS
    CHECKPOINT_PATH = "/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train/yolo26m_auair_final_scratch/weights/last.pt"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # Load the model
    model = YOLO(CHECKPOINT_PATH)
    
    print("🚀 STARTING PRECISION BOOST PHASE (Target: +1.5% mAP)")
    print("Strategy: Doubling Classification sensitivity to force Human/Motorbike detection.")

    # BOOSTED TRAINING PARAMS
    model.train(
        data="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/data/auair_norm.yaml",
        epochs=100,      # Continue for 65 more epochs (totaling 100)
        imgsz=800,
        batch=8,
        lr0=0.001,       # Lower starting LR for fine-tuning precision
        lrf=0.1,         # Final LR lower
        cls=1.5,         # INCREASED Classification weight (Default 0.5) - FOCUS ON CORRECT CLASS ID
        box=10.0,        # INCREASED Box weight (Default 7.5) - FOCUS ON SMALL OBJECT BOUNDARIES
        patience=30,
        project="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train",
        name="yolo26m_precison_boost_v3"
    )

if __name__ == "__main__":
    precision_boost()
