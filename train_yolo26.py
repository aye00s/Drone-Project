from ultralytics import YOLO
import os

def main():
    # Load the YOLOv26 medium model
    # Using the absolute path to the weights found in T02 directory
    model_path = "/home/students/bigdrive/Drone/T02_HIT_UAV_Det/yolo26m.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    model = YOLO(model_path)

    # Train the model
    # data: path to the dataset YAML (using absolute path)
    # epochs: number of training epochs
    # imgsz: input image size
    # batch: batch size
    results = model.train(
        data="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/data/auair.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        cos_lr=True,
        close_mosaic=10,
        name="yolo26m_auair_full",
        project="/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train"
    )

    print("Training complete. Results saved to runs/train/yolo26m_auair")

if __name__ == "__main__":
    main()
