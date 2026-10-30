"""
Train YOLOv8n-cls for Leaf Crop & Disease Classification.
Leverages deep CSPDarknet backbone, automatic Albumentations augmentations,
MixUp/Mosaic, and cross-entropy loss with cosine annealing.
"""

import sys
import time
from pathlib import Path
from ultralytics import YOLO

def main():
    data_dir = Path("data").resolve()
    print(f"Dataset directory: {data_dir}")
    
    # Load pretrained YOLOv8 nano classifier
    model = YOLO("yolov8n-cls.pt")
    
    epochs = 8
    imgsz = 224
    batch_size = 64
    
    print(f"Starting YOLOv8n-cls training for {epochs} epochs...")
    print(f"Image size: {imgsz}, Batch size: {batch_size}")
    
    t0 = time.time()
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        workers=2,
        device="cpu",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        patience=5,
        save=True,
        project="yolo_runs",
        name="leaf_cls",
        exist_ok=True,
        verbose=True
    )
    
    total_time = time.time() - t0
    print(f"Training finished in {total_time/60.0:.2f} minutes.")
    
    # Evaluate on test set
    print("Evaluating best model on test set...")
    test_metrics = model.val(data=str(data_dir), split="test", imgsz=imgsz, batch=batch_size)
    print(f"Test Top-1 Accuracy: {test_metrics.top1:.4f}")
    print(f"Test Top-5 Accuracy: {test_metrics.top5:.4f}")
    
    # Export to ONNX for lightning-fast CPU inference in FastAPI
    print("Exporting best model to ONNX...")
    best_pt = Path("yolo_runs/leaf_cls/weights/best.pt")
    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        exported_path = best_model.export(format="onnx", imgsz=imgsz, dynamic=False)
        print(f"ONNX exported successfully to: {exported_path}")
    else:
        exported_path = model.export(format="onnx", imgsz=imgsz, dynamic=False)
        print(f"Exported to: {exported_path}")

if __name__ == "__main__":
    main()
