from ultralytics import YOLO
import os

def main():
    # Path to the last checkpoint
    checkpoint_path = "/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train/yolo26m_auair_full/weights/last.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # Load the model from the checkpoint
    model = YOLO(checkpoint_path)

    # Resume the training
    # Note: When resuming, most arguments are loaded from the original run
    results = model.train(resume=True)

    print("Resumed training finished.")

if __name__ == "__main__":
    main()
