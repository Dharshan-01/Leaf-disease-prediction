"""
Evaluation and Diagnostics Script for Plant Disease Classifier.
- Loads best checkpoint (best_model.pth)
- Computes Test Accuracy and Loss across data/test/
- Computes Confusion Matrix and plots high-resolution heatmap
- Tracks per-class Precision, Recall, and F1-Score (classification report)
- Specifically analyzes visually similar blights and minority classes (Potato___healthy)
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from dataset import get_data_loaders
from model import create_model


def plot_confusion_matrix(cm, class_names, output_path="confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix: Plant Disease Classification (MobileNetV2)", fontsize=14, pad=15)
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    short_names = [c.replace("___", "\n").replace("__", "\n") for c in class_names]
    plt.xticks(tick_marks, short_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, short_names, fontsize=9)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=10, weight="bold")

    plt.ylabel("True Class Label", fontsize=11, labelpad=10)
    plt.xlabel("Predicted Class Label", fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {Path(output_path).resolve()}")


def evaluate(checkpoint_path="best_model.pth", data_dir="data", batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    # 1. Load Data
    loaders, class_names, class_weights = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=224,
        num_workers=0
    )
    test_loader = loaders["test"]
    print(f"Test samples: {len(test_loader.dataset)} across {len(class_names)} classes.")

    # 2. Load Checkpoint
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_file.resolve()}. Train the model first.")

    checkpoint = torch.load(checkpoint_file, map_location=device)
    model = create_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint saved from Epoch {checkpoint.get('epoch', 'N/A')} (Stage {checkpoint.get('stage', 'N/A')})")

    # 3. Predict on Test Set
    all_preds = []
    all_targets = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)

            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_loss = running_loss / len(all_targets)
    test_acc = (all_preds == all_targets).mean()

    print("\n" + "=" * 70)
    print(f"TEST EVALUATION RESULTS (Overall Accuracy: {test_acc*100:.2f}% | Loss: {test_loss:.4f})")
    print("=" * 70)

    # 4. Classification Report
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(all_targets, all_preds, target_names=class_names, digits=4)
    print(report_str)

    # 5. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(cm, class_names, "confusion_matrix.png")

    # 6. Specific Diagnostic Checks
    print("\n" + "=" * 70)
    print("DIAGNOSTIC HIGHLIGHTS:")
    print("=" * 70)

    # Potato healthy check
    if "Potato___healthy" in class_names:
        p_idx = class_names.index("Potato___healthy")
        p_metrics = report_dict["Potato___healthy"]
        print(f"1. Minority Class Analysis (Potato___healthy):")
        print(f"   Precision: {p_metrics['precision']*100:.2f}% | Recall: {p_metrics['recall']*100:.2f}% | F1: {p_metrics['f1-score']*100:.2f}% (Support: {p_metrics['support']})")

    # Blight confusion check
    pairs = [
        ("Potato___Early_blight", "Potato___Late_blight"),
        ("Tomato_Early_blight", "Tomato_Late_blight")
    ]
    for c1, c2 in pairs:
        if c1 in class_names and c2 in class_names:
            i1, i2 = class_names.index(c1), class_names.index(c2)
            c1_to_c2 = cm[i1, i2]
            c2_to_c1 = cm[i2, i1]
            print(f"2. Blight Confusion ({c1} vs {c2}):")
            print(f"   - {c1} misclassified as {c2}: {c1_to_c2} instances")
            print(f"   - {c2} misclassified as {c1}: {c2_to_c1} instances")

    # Save summary
    results = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }
    with open("evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nEvaluation report saved to: evaluation_report.json")
    return results


if __name__ == "__main__":
    evaluate()
