from ultralytics import YOLO
import os

def resume_master():
    # PATH TO THE LATEST CHECKPOINT
    CHECKPOINT_PATH = "/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train/yolo26m_auair_final_scratch/weights/last.pt"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # Load and resume
    model = YOLO(CHECKPOINT_PATH)
    
    print(f"🚀 Resuming Master Scratch Run from {CHECKPOINT_PATH}")
    
    # Official Resume command
    model.train(resume=True)

if __name__ == "__main__":
    resume_master()
