from ultralytics import YOLO
import os

def ultimate_accuracy_push():
    # PATH TO OUR BEST-PERFORMING CHECKPOINT
    CHECKPOINT_PATH = "/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train/yolo26m_auair_final_scratch/weights/last.pt"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: {CHECKPOINT_PATH} not found.")
        return

    # Load the high-quality model
    model = YOLO(CHECKPOINT_PATH)
    
    print("🌟 STARTING ULTIMATE ACCURACY PHASE (1024px Resolution)")
    print("This sequence is designed to shatter the 36.3% barrier by capturing tiny object details.")

    # ULTIMATE SETTINGS
    model.train(
        data="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/data/auair_norm.yaml",
        epochs=100,
        imgsz=1024,      # RESOLUTION BOOST - THE GAME CHANGER
        batch=4,         # Lowered batch to handle 1024px memory load
        lr0=0.001,       # Low LR to allow high-res feature integration
        lrf=0.1,
        patience=50,
        close_mosaic=10, # Sharpen detections in final 10 epochs
        project="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train",
        name="yolo26m_auair_1024_ultimate"
    )

if __name__ == "__main__":
    ultimate_accuracy_push()
