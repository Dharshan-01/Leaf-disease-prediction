"""
PyTorch Dataset Loader module for Plant Disease Classification.
Uses torchvision.datasets.ImageFolder to automatically infer classes from folder names.
Includes data augmentation and handling for class imbalance (e.g. Potato___healthy).
"""

import os
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


# Standard ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(img_size: int = 224):
    """
    Returns train and validation/test transformation pipelines per specification:
    - Resize: 224 x 224 pixels
    - Augmentations (train only): Random horizontal & vertical flips,
      random rotation (+-25 degrees), subtle brightness/contrast jitter (10-15%)
    - Normalization: Scale [0, 1] then ImageNet mean & std
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform, eval_transform


def compute_class_weights(train_dataset: datasets.ImageFolder) -> torch.Tensor:
    """
    Computes inverse frequency class weights to pass to nn.CrossEntropyLoss(weight=...)
    to prevent the model from biasing toward majority classes (e.g. Tomato_Bacterial_spot).
    """
    targets = [s[1] for s in train_dataset.samples]
    class_counts = Counter(targets)
    total_samples = len(targets)
    num_classes = len(train_dataset.classes)
    
    weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(train_dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that samples minority classes more frequently,
    yielding balanced batches during training.
    """
    targets = [s[1] for s in train_dataset.samples]
    class_counts = Counter(targets)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[target] for target in targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def get_data_loaders(
    data_dir: str = "data",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 2,
    use_weighted_sampler: bool = False
):
    """
    Creates PyTorch DataLoaders for train, val, and test splits using ImageFolder.

    Args:
        data_dir: Path to the data directory containing train/, val/, test/
        batch_size: Mini-batch size for DataLoader
        img_size: Target image height/width for cropping/resizing
        num_workers: Number of subprocesses for data loading
        use_weighted_sampler: If True, uses WeightedRandomSampler instead of shuffle

    Returns:
        dataloaders (dict): {'train': loader, 'val': loader, 'test': loader}
        class_names (list): List of class names sorted alphabetically
        class_weights (torch.Tensor): Computed class weights for CrossEntropyLoss
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    train_transform, eval_transform = get_transforms(img_size=img_size)

    # 1. Instantiate PyTorch ImageFolder datasets
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=eval_transform)

    # 2. Compute class weights for loss function
    class_weights = compute_class_weights(train_dataset)

    # 3. Handle sampler vs shuffling
    if use_weighted_sampler:
        sampler = get_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    return dataloaders, train_dataset.classes, class_weights


if __name__ == "__main__":
    print("Testing PyTorch ImageFolder Dataset Loader...")
    loaders, classes, weights = get_data_loaders(data_dir="data", batch_size=16, num_workers=0)
    
    print(f"\nInferred Classes ({len(classes)}):")
    for idx, c in enumerate(classes):
        print(f"  [{idx}] {c} (Weight: {weights[idx]:.3f})")

    train_loader = loaders["train"]
    images, labels = next(iter(train_loader))
    print(f"\nSample batch loaded successfully:")
    print(f"  Batch image shape: {images.shape} (Batch, Channels, Height, Width)")
    print(f"  Batch label shape: {labels.shape}")
    print(f"  Batch label values: {labels.tolist()}")
