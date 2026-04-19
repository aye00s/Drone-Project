from ultralytics import YOLO
import os

def turbo_hd_push():
    # PATHS
    CHECKPOINT_PATH = "/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train/yolo26m_auair_final_scratch/weights/last.pt"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found.")
        return

    # Load model
    model = YOLO(CHECKPOINT_PATH)
    
    print("🚀 LAUNCHING TURBO-HD (1024px with Rectangular Efficiency)")
    print("Goal: Reach 36.3% mAP50 within the next 8-12 hours.")

    # TURBO OPTIMIZED SETTINGS
    model.train(
        data="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/data/auair_norm.yaml",
        epochs=100,
        imgsz=1024,
        batch=16,        # QUADRUPLED batch size for better GPU utilization
        rect=True,       # ENABLE RECTANGULAR TRAINING (1.7x Speedup for AU-AIR aspect ratio)
        lr0=0.001,       # Precise fine-tuning
        workers=8,
        project="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train",
        name="yolo26m_auair_turbo_hd"
    )

if __name__ == "__main__":
    turbo_hd_push()
