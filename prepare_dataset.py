import os
import shutil
import zipfile
import random
from collections import Counter
from pathlib import Path

# Configuration
ZIP_PATH = Path("dataset.zip")
RAW_DIR = Path("PlantVillage")
EXCLUDED_DIR = Path("PlantVillage_excluded")
DATA_DIR = Path("data")
RANDOM_SEED = 42

# Target Classes (9 selected classes)
TARGET_CLASSES = [
    # Pepper (2 classes)
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    # Potato (3 classes)
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    # Tomato (4 selected key classes)
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


def step1_extract_zip():
    print("=" * 60)
    print("STEP 1: Extracting target directories from ZIP archive...")
    print("=" * 60)
    
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Zip archive not found at: {ZIP_PATH.resolve()}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        # Avoid the duplicate 'plantvillage/PlantVillage/' entries in the zip
        # and extract only the canonical 'PlantVillage/' files
        members = [m for m in z.namelist() if m.startswith("PlantVillage/") and not m.endswith("/")]
        print(f"Total canonical files found in archive: {len(members)}")
        
        extracted_count = 0
        for member in members:
            # member format: PlantVillage/<class_name>/<filename>
            parts = member.split('/')
            if len(parts) >= 3:
                class_name = parts[1]
                filename = parts[-1]
                
                target_folder = RAW_DIR / class_name
                target_folder.mkdir(parents=True, exist_ok=True)
                target_file = target_folder / filename
                
                if not target_file.exists():
                    with z.open(member) as source, open(target_file, "wb") as dest:
                        shutil.copyfileobj(source, dest)
                    extracted_count += 1
                    
        print(f"Extracted {extracted_count} images into '{RAW_DIR}'.")


def step2_filter_classes():
    print("\n" + "=" * 60)
    print("STEP 2: Selecting target classes and isolating excluded folders...")
    print("=" * 60)
    
    EXCLUDED_DIR.mkdir(parents=True, exist_ok=True)
    all_dirs = [d for d in RAW_DIR.iterdir() if d.is_dir()]
    
    kept_count = 0
    moved_count = 0
    
    for d in all_dirs:
        if d.name in TARGET_CLASSES:
            kept_count += 1
            print(f"  [KEEP] {d.name} ({len(list(d.glob('*.*')))} images)")
        else:
            dest = EXCLUDED_DIR / d.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(d), str(dest))
            moved_count += 1
            print(f"  [MOVED AWAY] {d.name} -> moved to '{EXCLUDED_DIR.name}/'")
            
    print(f"\nResult: Kept {kept_count} target classes. Moved {moved_count} excluded classes to '{EXCLUDED_DIR.name}/'.")


def step3_split_dataset():
    print("\n" + "=" * 60)
    print("STEP 3: Splitting data into Train (70%), Val (20%), and Test (10%)...")
    print("=" * 60)
    
    rng = random.Random(RANDOM_SEED)
    splits = ["train", "val", "test"]
    
    for s in splits:
        for c in TARGET_CLASSES:
            (DATA_DIR / s / c).mkdir(parents=True, exist_ok=True)
            
    stats = {}
    
    for class_name in TARGET_CLASSES:
        source_class_dir = RAW_DIR / class_name
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
        images = [f for f in source_class_dir.iterdir() if f.is_file() and f.suffix in valid_extensions]
        
        # Deterministic shuffle
        rng.shuffle(images)
        total = len(images)
        
        n_train = int(round(total * TRAIN_RATIO))
        n_val = int(round(total * VAL_RATIO))
        n_test = total - n_train - n_val
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        for img in train_imgs:
            shutil.copy2(img, DATA_DIR / "train" / class_name / img.name)
        for img in val_imgs:
            shutil.copy2(img, DATA_DIR / "val" / class_name / img.name)
        for img in test_imgs:
            shutil.copy2(img, DATA_DIR / "test" / class_name / img.name)
            
        stats[class_name] = {
            "total": total,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs)
        }
        
    return stats


def step4_verify_balance(stats):
    print("\n" + "=" * 60)
    print("STEP 4: Class Balance Verification & Analysis")
    print("=" * 60)
    
    header = f"{'Class Name':<40} | {'Total':>6} | {'Train':>6} | {'Val':>6} | {'Test':>6} | {'% of Total':>10}"
    print(header)
    print("-" * len(header))
    
    grand_total = sum(v["total"] for v in stats.values())
    grand_train = sum(v["train"] for v in stats.values())
    grand_val = sum(v["val"] for v in stats.values())
    grand_test = sum(v["test"] for v in stats.values())
    
    imbalance_notes = []
    
    for cls, counts in sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True):
        pct = (counts["total"] / grand_total) * 100
        print(f"{cls:<40} | {counts['total']:>6} | {counts['train']:>6} | {counts['val']:>6} | {counts['test']:>6} | {pct:>9.2f}%")
        if counts["total"] < 500:
            imbalance_notes.append((cls, counts["total"]))
            
    print("-" * len(header))
    print(f"{'TOTAL':<40} | {grand_total:>6} | {grand_train:>6} | {grand_val:>6} | {grand_test:>6} | {100.0:>9.2f}%")
    
    print("\n[!] Class Imbalance Highlights:")
    for cls, count in imbalance_notes:
        print(f"  * MINORITY CLASS: '{cls}' has only {count} images ({count/grand_total*100:.2f}% of dataset).")
    print("  -> Recommendation: Apply strong data augmentation (random flips, rotations, color jitter) or class-weighted loss.")


if __name__ == "__main__":
    step1_extract_zip()
    step2_filter_classes()
    stats = step3_split_dataset()
    step4_verify_balance(stats)
    print("\nAll steps completed successfully!")