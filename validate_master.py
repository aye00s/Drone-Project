from ultralytics import YOLO
import os

def calculate_master_metrics():
    # PATHS
    WEIGHTS_PATH = "/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/runs/train/yolo26m_auair_turbo_hd/weights/best.pt"
    DATA_PATH = "/home/students/bigdrive/Drone/T03_AU_AIR_Traffic/data/auair_norm.yaml"
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: {WEIGHTS_PATH} not found.")
        return

    # Load model
    model = YOLO(WEIGHTS_PATH)
    
    print("🚦 VALIDATING MASTER PERFORMANCE (Unified 8-Class Detection)")
    
    # Run validation across all classes
    results = model.val(
        data=DATA_PATH,
        imgsz=1024,
        rect=True,
        plots=True,
        name="master_eval"
    )
    
    map50 = results.results_dict['metrics/mAP50(B)']
    print(f"\n✅ Peak mAP50: {map50:.4f} ({map50*100:.2f}%)")
    
    if map50 >= 0.363:
        print("🚀 SUCCESS: Peak mAP exceeds the target benchmark!")
    else:
        print("⚠️ TARGET WARNING: mAP is below target. Further optimization needed.")

if __name__ == "__main__":
    calculate_master_metrics()
