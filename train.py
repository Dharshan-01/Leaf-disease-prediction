"""
High-Accuracy EfficientNet-B0 Training Pipeline for Plant Disease Classification.

Key Innovations:
1. Backbone: EfficientNet-B0 with Squeeze-and-Excitation (SE) channel attention
   specifically tuned for fine-grained leaf lesion textures and chlorosis rings.
2. Multi-Pass Feature Augmentation: Passes images through backbone multiple times
   with dynamic RandomResizedCrop, Rotation, Flips, and ColorJitter, multiplying
   effective training diversity without slowing down head convergence.
3. Single-Weighting Balanced Sampler: Zero double-weighting artifacts.
4. Top-Stages Fine-Tuning: End-to-end alignment of high-level semantic representation.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from dataset import get_data_loaders
from model import create_model

# Optimize CPU threads for 12-core system
if hasattr(os, "cpu_count"):
    threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(threads)


def extract_augmented_features(model, loader, device, num_passes=2, desc="Extracting"):
    """
    Passes images through frozen EfficientNet-B0 backbone.
    When num_passes > 1, each pass uses different dynamic random crops, flips,
    and lighting jitter from train_transform, effectively multiplying the training data!
    """
    model.eval()
    all_features = []
    all_labels = []

    total_batches = len(loader) * num_passes
    curr_batch = 0

    with torch.no_grad():
        for p in range(num_passes):
            for i, (images, labels) in enumerate(loader):
                images = images.to(device)
                feats = model.extract_features(images)
                all_features.append(feats.cpu())
                all_labels.append(labels)
                curr_batch += 1

                if curr_batch % 40 == 0 or curr_batch == total_batches:
                    pct = curr_batch / total_batches * 100
                    print(f"  {desc} (pass {p+1}/{num_passes}): {curr_batch}/{total_batches} batches ({pct:.1f}%)", flush=True)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print(f"  Extracted {features.shape[0]} diverse feature vectors of dim {features.shape[1]}")
    return features, labels


def train_head_on_features(head, train_features, train_labels, val_features, val_labels,
                           device, epochs=35, lr=1e-3, batch_size=128):
    """
    Train classifier head cleanly using a balanced sampler WITHOUT double weighting.
    """
    train_ds = TensorDataset(train_features, train_labels)

    # Balanced sampling across minority and majority classes
    targets = train_labels.numpy()
    class_counts = Counter(targets)
    sample_weights = [1.0 / class_counts[int(t)] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_ds = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_head_state = None
    patience, counter = 8, 0

    for epoch in range(1, epochs + 1):
        head.train()
        running_loss, correct, total = 0.0, 0, 0
        for feats, labs in train_loader:
            feats, labs = feats.to(device), labs.to(device)
            optimizer.zero_grad()
            out = head(feats)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * feats.size(0)
            correct += out.argmax(1).eq(labs).sum().item()
            total += labs.size(0)

        # Validate
        head.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for feats, labs in val_loader:
                feats, labs = feats.to(device), labs.to(device)
                out = head(feats)
                v_loss += criterion(out, labs).item() * feats.size(0)
                v_corr += out.argmax(1).eq(labs).sum().item()
                v_tot += labs.size(0)

        val_loss = v_loss / v_tot
        val_acc = v_corr / v_tot
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_head_state = {k: v.clone() for k, v in head.state_dict().items()}
            counter = 0
            flag = " *BEST*"
        else:
            counter += 1
            flag = ""

        print(f"Epoch {epoch:02d}/{epochs} - Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%{flag}")
        if counter >= patience:
            print("Early stopping triggered.")
            break

    return best_head_state, best_val_loss


def finetune_full_model(model, train_loader, val_loader, class_weights, device,
                        epochs=3, lr_backbone=8e-6, lr_head=3e-5):
    """
    End-to-end fine-tuning with top stages of EfficientNet-B0 unfrozen.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)

    model.unfreeze_top_layers(num_blocks=3)
    param_groups = model.get_parameter_groups(lr_backbone=lr_backbone, lr_head=lr_head)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(train_loader)
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            if (i + 1) % 50 == 0:
                print(f"    Batch {i+1}/{num_batches} ...", flush=True)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                out = model(images)
                loss = criterion(out, labels)
                val_loss_sum += loss.item() * images.size(0)
                val_correct += out.argmax(1).eq(labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            flag = " *BEST*"
        else:
            flag = ""

        elapsed = time.time() - t0
        print(f"  FT Epoch {epoch}/{epochs} [{elapsed:.0f}s] "
              f"Train: {train_loss:.4f} / {train_acc*100:.1f}% | "
              f"Val: {val_loss:.4f} / {val_acc*100:.1f}%{flag}", flush=True)

    return best_state, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-epochs", type=int, default=35)
    parser.add_argument("--ft-epochs", type=int, default=3)
    parser.add_argument("--passes", type=int, default=2,
                        help="Number of multi-crop augmentation passes through backbone")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default="best_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} (threads={torch.get_num_threads()})")
    print(f"Backbone: EfficientNet-B0 with SE Attention")
    print(f"Strategy: {args.passes}-Pass Augmented Extraction -> Head ({args.head_epochs}ep) -> Fine-Tune ({args.ft_epochs}ep)")

    # 1. Load data
    loaders, class_names, class_weights = get_data_loaders(
        data_dir="data", batch_size=args.batch_size, img_size=224, num_workers=0
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train: {len(loaders['train'].dataset)} | Val: {len(loaders['val'].dataset)} | Test: {len(loaders['test'].dataset)}")

    # 2. Build model with pretrained EfficientNet-B0
    model = create_model(num_classes=num_classes, dropout_rate=0.3, pretrained=True).to(device)
    model.freeze_backbone()

    # ============================================================
    # PHASE 0: Multi-Pass Augmented Feature Extraction
    # ============================================================
    print("\n" + "=" * 65)
    print(f"PHASE 0: Extracting {args.passes}-Pass Augmented Features from EfficientNet-B0")
    print("=" * 65)

    t0 = time.time()
    train_features, train_labels = extract_augmented_features(
        model, loaders["train"], device, num_passes=args.passes, desc="Train features"
    )
    val_features, val_labels = extract_augmented_features(
        model, loaders["val"], device, num_passes=1, desc="Val features"
    )
    print(f"Feature extraction completed in {time.time()-t0:.1f}s")

    # ============================================================
    # PHASE 1: Train classifier head on diverse augmented features
    # ============================================================
    print("\n" + "=" * 65)
    print(f"PHASE 1: Training Classifier Head ({args.head_epochs} epochs on cached features)")
    print("=" * 65)

    head = model.backbone.classifier.to(device)

    t0 = time.time()
    best_head_state, best_head_loss = train_head_on_features(
        head, train_features, train_labels, val_features, val_labels,
        device, epochs=args.head_epochs, lr=1e-3, batch_size=128
    )
    print(f"Head training completed in {time.time()-t0:.1f}s")

    model.backbone.classifier.load_state_dict(best_head_state)

    # Save head-only checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "stage": "head_only",
        "val_loss": best_head_loss
    }, args.checkpoint)
    print(f"Head-only checkpoint saved: {Path(args.checkpoint).resolve()}")

    # ============================================================
    # PHASE 2: End-to-end fine-tuning
    # ============================================================
    if args.ft_epochs > 0:
        print("\n" + "=" * 65)
        print(f"PHASE 2: End-to-End Fine-Tuning ({args.ft_epochs} epochs, top 3 stages unfrozen)")
        print("=" * 65)

        t0 = time.time()
        best_ft_state, best_ft_loss = finetune_full_model(
            model, loaders["train"], loaders["val"], class_weights, device,
            epochs=args.ft_epochs, lr_backbone=8e-6, lr_head=3e-5
        )
        print(f"Fine-tuning completed in {time.time()-t0:.1f}s")

        if best_ft_state is not None:
            model.load_state_dict(best_ft_state)
            torch.save({
                "model_state_dict": best_ft_state,
                "class_names": class_names,
                "stage": "finetuned",
                "val_loss": best_ft_loss
            }, args.checkpoint)
            print(f"Fine-tuned checkpoint saved: {Path(args.checkpoint).resolve()}")

    # ============================================================
    # Test evaluation
    # ============================================================
    print("\n" + "=" * 65)
    print("QUICK TEST EVALUATION")
    print("=" * 65)
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in loaders["test"]:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            test_correct += preds.eq(labels).sum().item()
            test_total += labels.size(0)
    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc*100:.2f}% ({test_correct}/{test_total})")
    print(f"\nAll training completed successfully!")


if __name__ == "__main__":
    main()
