from ultralytics import YOLO
import os

def resume_stable():
    # PATH TO OUR BEST-PERFORMING CHECKPOINT
    CHECKPOINT_PATH = "/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train/yolo26m_auair_final_scratch/weights/last.pt"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found.")
        return

    # Load the high-quality model
    model = YOLO(CHECKPOINT_PATH)
    
    print("💎 RESUMING STABLE MASTER RUN (Restoring standard weights for final 36% push)")

    # STABLE FINISH - 100 Epochs total
    model.train(
        data="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/data/auair_norm.yaml",
        epochs=100,
        imgsz=800,
        batch=8,
        lr0=0.01,         # Standard starting point for healthy learning
        close_mosaic=20,  # Turn off mosaics near the end to sharpen boundaries
        project="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train",
        name="yolo26m_auair_final_scratch",
        resume=True       # PICK UP EXACTLY AT EPOCH 33
    )

if __name__ == "__main__":
    resume_stable()
