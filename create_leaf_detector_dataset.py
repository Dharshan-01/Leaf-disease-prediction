"""
Synthetic Background & Scale Augmentation Dataset Generator
for YOLOv8 Leaf Object Detection (Pepper, Potato, Tomato).
Creates realistic training samples of leaves placed on:
- Wooden tables & countertops
- Hand / finger hold tones
- Soil, gravel, garden, and neutral surfaces
"""

import os
import random
import glob
from pathlib import Path
import cv2
import numpy as np

def create_background(bg_type: str, size=(640, 640)) -> np.ndarray:
    w, h = size
    if bg_type == "wood":
        # Realistic wood grain texture simulation
        # Base brown color (R: 120-180, G: 70-110, B: 30-70)
        base_r = random.randint(120, 180)
        base_g = random.randint(70, 110)
        base_b = random.randint(30, 70)
        img = np.full((h, w, 3), [base_b, base_g, base_r], dtype=np.uint8)
        # Add grain lines
        noise = np.random.normal(0, 12, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (1, 15), 0)
        img = np.clip(img.astype(np.float32) + noise[:, :, None], 0, 255).astype(np.uint8)
        return img
    elif bg_type == "hand_skin":
        # Skin / hand tone (R: 180-230, G: 140-180, B: 110-150)
        base_r = random.randint(180, 230)
        base_g = random.randint(140, 180)
        base_b = random.randint(110, 150)
        img = np.full((h, w, 3), [base_b, base_g, base_r], dtype=np.uint8)
        noise = np.random.normal(0, 6, (h, w, 3)).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif bg_type == "soil":
        # Dark garden soil / earth (R: 50-80, G: 40-70, B: 30-60)
        base_r = random.randint(50, 80)
        base_g = random.randint(40, 70)
        base_b = random.randint(30, 60)
        img = np.full((h, w, 3), [base_b, base_g, base_r], dtype=np.uint8)
        noise = np.random.normal(0, 18, (h, w, 3)).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif bg_type == "tabletop":
        # Kitchen counter / tile / white-gray desk
        val = random.randint(180, 240)
        img = np.full((h, w, 3), [val, val, val], dtype=np.uint8)
        noise = np.random.normal(0, 8, (h, w, 3)).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    else:
        # Neutral background
        val = random.randint(100, 180)
        return np.full((h, w, 3), [val, val, val], dtype=np.uint8)

def extract_clean_leaf(img_bgr: np.ndarray):
    """Isolates the leaf foreground mask cleanly."""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Check if corners are mostly inverted
    corners = [thresh[0,0], thresh[0,-1], thresh[-1,0], thresh[-1,-1]]
    if sum(corners) > 2 * 255:
        thresh = cv2.bitwise_not(thresh)
        
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    exg = 2.0 * g - r - b
    exg_mask = (exg > -5.0).astype(np.uint8) * 255
    combined = cv2.bitwise_or(thresh, exg_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        mask = np.ones((h, w), dtype=np.uint8) * 255
        return img_bgr, mask, (0, 0, w, h)
        
    min_area = (h * w) * 0.02
    valid = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if valid:
        hull = cv2.convexHull(np.vstack(valid))
    else:
        hull = max(cnts, key=cv2.contourArea)
        
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)
    bx, by, bw, bh = cv2.boundingRect(hull)
    return img_bgr, mask, (bx, by, bw, bh)

def main():
    out_dir = Path("leaf_detector_dataset")
    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    class_map = {
        "Pepper": 0,
        "Potato": 1,
        "Tomato": 2
    }
    
    bg_types = ["wood", "hand_skin", "soil", "tabletop", "neutral"]
    
    # Collect source images
    dataset_paths = {
        "Pepper": glob.glob("data/train/Pepper*/*.JPG") + glob.glob("data/train/Pepper*/*.jpg"),
        "Potato": glob.glob("data/train/Potato*/*.JPG") + glob.glob("data/train/Potato*/*.jpg"),
        "Tomato": glob.glob("data/train/Tomato*/*.JPG") + glob.glob("data/train/Tomato*/*.jpg")
    }
    
    samples_per_crop_train = 250  # 750 train samples
    samples_per_crop_val = 50     # 150 val samples
    
    canvas_size = 640
    
    for split, count in [("train", samples_per_crop_train), ("val", samples_per_crop_val)]:
        idx = 0
        for crop_name, cls_id in class_map.items():
            imgs = dataset_paths[crop_name]
            chosen = random.sample(imgs, min(count, len(imgs)))
            for p in chosen:
                raw_img = cv2.imread(p)
                if raw_img is None: continue
                
                # Extract clean leaf cutout
                img_bgr, mask, (bx, by, bw, bh) = extract_clean_leaf(raw_img)
                leaf_crop = img_bgr[by:by+bh, bx:bx+bw]
                mask_crop = mask[by:by+bh, bx:bx+bw]
                
                # Create a random background canvas
                bg_t = random.choice(bg_types)
                canvas = create_background(bg_t, size=(canvas_size, canvas_size))
                
                # Random scale for leaf: 0.35 to 0.85 of canvas
                target_scale = random.uniform(0.35, 0.82)
                max_leaf_dim = max(bw, bh)
                target_dim = int(canvas_size * target_scale)
                scale_ratio = target_dim / max(max_leaf_dim, 1)
                new_w = max(10, int(bw * scale_ratio))
                new_h = max(10, int(bh * scale_ratio))
                
                leaf_resized = cv2.resize(leaf_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
                # Random position
                max_x = max(0, canvas_size - new_w)
                max_y = max(0, canvas_size - new_h)
                pos_x = random.randint(0, max_x)
                pos_y = random.randint(0, max_y)
                
                # Alpha composite leaf onto background canvas
                alpha = (mask_resized.astype(np.float32) / 255.0)[:, :, None]
                roi = canvas[pos_y:pos_y+new_h, pos_x:pos_x+new_w].astype(np.float32)
                composite = roi * (1.0 - alpha) + leaf_resized.astype(np.float32) * alpha
                canvas[pos_y:pos_y+new_h, pos_x:pos_x+new_w] = composite.astype(np.uint8)
                
                # Compute YOLO normalized bounding box: cx, cy, w, h
                cx = (pos_x + new_w / 2.0) / canvas_size
                cy = (pos_y + new_h / 2.0) / canvas_size
                nw = new_w / float(canvas_size)
                nh = new_h / float(canvas_size)
                
                # Save image and label
                sample_name = f"{crop_name}_{split}_{idx:04d}"
                img_out_path = out_dir / "images" / split / f"{sample_name}.jpg"
                lbl_out_path = out_dir / "labels" / split / f"{sample_name}.txt"
                
                cv2.imwrite(str(img_out_path), canvas)
                with open(lbl_out_path, "w") as f:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    
                idx += 1
                
        print(f"Generated {split} split with {idx} images.")
        
    # Write dataset.yaml
    yaml_content = f"""
path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val
names:
  0: pepper_leaf
  1: potato_leaf
  2: tomato_leaf
"""
    with open(out_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content.strip())
        
    print(f"Dataset configuration written to {out_dir / 'dataset.yaml'}")

if __name__ == "__main__":
    main()
