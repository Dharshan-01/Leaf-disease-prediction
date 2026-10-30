"""
Fine-tune YOLOv8 Object Detector (yolov8n.pt) to detect:
0: pepper_leaf
1: potato_leaf
2: tomato_leaf
Across arbitrary real-world backgrounds (wood, skin/hands, soil, tabletop).
"""

import time
from pathlib import Path
from ultralytics import YOLO

def main():
    data_yaml = Path("leaf_detector_dataset/dataset.yaml").resolve()
    print(f"Loading dataset from: {data_yaml}")
    
    # Load pretrained YOLOv8 nano object detector
    model = YOLO("yolov8n.pt")
    
    epochs = 12
    imgsz = 512
    batch = 16
    
    print(f"Training YOLOv8 Leaf Detector for {epochs} epochs...")
    t0 = time.time()
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2,
        device="cpu",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.002,
        patience=5,
        save=True,
        project="runs/detect",
        name="leaf_detector",
        exist_ok=True,
        verbose=True
    )
    
    print(f"Leaf detector training complete in {(time.time()-t0)/60.0:.2f} minutes.")
    
    # Validate best model
    val_metrics = model.val(data=str(data_yaml), imgsz=imgsz, batch=batch)
    print(f"Validation mAP50: {val_metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {val_metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
