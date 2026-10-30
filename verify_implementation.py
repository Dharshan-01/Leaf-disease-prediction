"""
Verification script for full-leaf localization and accurate crop pathology.
Tests:
1. User's exact Tomato leaf (test_user_tomato.png)
2. Bell Pepper leaf sample
3. Potato leaf sample
4. Non-target / unsupported input
"""

import sys
import glob
import cv2
import numpy as np
from PIL import Image
from leaf_analyzer import analyze_leaf_morphology, locate_full_leaf
from app import predict_leaf

def test_user_tomato():
    print("\n--- TEST CASE 1: User's Exact Tomato Leaf ---")
    img_path = "test_user_tomato.png"
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f"Could not read {img_path}"
    h, w = img_bgr.shape[:2]

    contour, leaf_mask, (bx, by, bw, bh), detected_by_yolo, conf = locate_full_leaf(img_bgr)
    coverage_w = bw / float(w)
    coverage_h = bh / float(h)
    print(f"Image shape: {w}x{h}")
    print(f"Bounding Box: x={bx}, y={by}, w={bw}, h={bh}")
    print(f"Width coverage: {coverage_w * 100:.1f}% | Height coverage: {coverage_h * 100:.1f}%")
    print(f"YOLO detected: {detected_by_yolo} (conf: {conf:.2f})")

    assert coverage_w >= 0.90, f"Expected bounding box to span >=90% of leaf width, got {coverage_w*100:.1f}%"
    print("PASS: Bounding box spans entire leaf width without truncation.")

    with open(img_path, "rb") as f:
        bytes_data = f.read()

    # Call morphology analysis
    morph = analyze_leaf_morphology(bytes_data)
    print(f"Solidity: {morph['morphology'].get('solidity')}, Circularity: {morph['morphology'].get('circularity')}")
    assert "solidity" in morph["morphology"] and "circularity" in morph["morphology"]

    # Test prediction
    from app import ort_session, ort_input_name, preprocess_crops, softmax, CLASS_NAMES, yolo_model, PLANT_DISEASE_DB
    leaf_bytes = morph.get("clean_leaf_bytes", bytes_data)
    
    # YOLO
    pil_img = Image.open(img_path).convert("RGB")
    yolo_res = yolo_model(pil_img, verbose=False)[0]
    y_dict = {yolo_res.names[i]: float(yolo_res.probs.data[i]) for i in range(len(yolo_res.names))}
    yolo_probs = np.array([y_dict.get(c, 0.0) for c in CLASS_NAMES], dtype=np.float32)
    
    # ONNX
    input_batch = preprocess_crops(leaf_bytes)
    outputs = ort_session.run(None, {ort_input_name: input_batch})
    onnx_probs = np.mean(softmax(outputs[0]), axis=0)

    probs = 0.70 * yolo_probs + 0.30 * onnx_probs
    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_crop = PLANT_DISEASE_DB[top_class]["plant"]

    print(f"Top Class: {top_class} ({probs[top_idx]*100:.1f}%)")
    print(f"Identified Crop: {top_crop}")

    assert top_crop == "Tomato", f"Expected Tomato crop, got {top_crop}"
    print("PASS: Identified crop is Tomato (not misclassified as Bell Pepper)!")


def test_bell_pepper():
    print("\n--- TEST CASE 2: Bell Pepper Leaf ---")
    files = glob.glob("data/test/Pepper__bell*/*.JPG") + glob.glob("data/test/Pepper__bell*/*.jpg")
    assert files, "No pepper test files found"
    img_path = files[0]
    with open(img_path, "rb") as f:
        bytes_data = f.read()

    morph = analyze_leaf_morphology(bytes_data)
    from app import ort_session, ort_input_name, preprocess_crops, softmax, CLASS_NAMES, yolo_model, PLANT_DISEASE_DB
    leaf_bytes = morph.get("clean_leaf_bytes", bytes_data)
    
    pil_img = Image.open(img_path).convert("RGB")
    yolo_res = yolo_model(pil_img, verbose=False)[0]
    y_dict = {yolo_res.names[i]: float(yolo_res.probs.data[i]) for i in range(len(yolo_res.names))}
    yolo_probs = np.array([y_dict.get(c, 0.0) for c in CLASS_NAMES], dtype=np.float32)
    
    input_batch = preprocess_crops(leaf_bytes)
    outputs = ort_session.run(None, {ort_input_name: input_batch})
    onnx_probs = np.mean(softmax(outputs[0]), axis=0)

    probs = 0.70 * yolo_probs + 0.30 * onnx_probs
    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_crop = PLANT_DISEASE_DB[top_class]["plant"]

    print(f"File: {img_path}")
    print(f"Top Class: {top_class} ({probs[top_idx]*100:.1f}%)")
    print(f"Identified Crop: {top_crop}")
    assert top_crop == "Bell Pepper", f"Expected Bell Pepper, got {top_crop}"
    print("PASS: Bell Pepper correctly identified.")


def test_potato():
    print("\n--- TEST CASE 3: Potato Leaf ---")
    files = glob.glob("data/test/Potato*/*.JPG") + glob.glob("data/test/Potato*/*.jpg")
    assert files, "No potato test files found"
    img_path = files[0]
    with open(img_path, "rb") as f:
        bytes_data = f.read()

    morph = analyze_leaf_morphology(bytes_data)
    from app import ort_session, ort_input_name, preprocess_crops, softmax, CLASS_NAMES, yolo_model, PLANT_DISEASE_DB
    leaf_bytes = morph.get("clean_leaf_bytes", bytes_data)
    
    pil_img = Image.open(img_path).convert("RGB")
    yolo_res = yolo_model(pil_img, verbose=False)[0]
    y_dict = {yolo_res.names[i]: float(yolo_res.probs.data[i]) for i in range(len(yolo_res.names))}
    yolo_probs = np.array([y_dict.get(c, 0.0) for c in CLASS_NAMES], dtype=np.float32)
    
    input_batch = preprocess_crops(leaf_bytes)
    outputs = ort_session.run(None, {ort_input_name: input_batch})
    onnx_probs = np.mean(softmax(outputs[0]), axis=0)

    probs = 0.70 * yolo_probs + 0.30 * onnx_probs
    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]
    top_crop = PLANT_DISEASE_DB[top_class]["plant"]

    print(f"File: {img_path}")
    print(f"Top Class: {top_class} ({probs[top_idx]*100:.1f}%)")
    print(f"Identified Crop: {top_crop}")
    assert top_crop == "Potato", f"Expected Potato, got {top_crop}"
    print("PASS: Potato correctly identified.")


def test_non_target():
    print("\n--- TEST CASE 4: Unsupported Non-Target Foliage & Non-Plant ---")
    # 4A: Empty / non-plant image (desk or wall)
    non_target_img = np.full((300, 300, 3), (200, 200, 200), dtype=np.uint8)
    cv2.putText(non_target_img, "TABLE SURFACE", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    _, buf = cv2.imencode(".jpg", non_target_img)

    morph = analyze_leaf_morphology(buf.tobytes())
    print(f"[Non-Plant Test] Is unsupported crop: {morph['is_unsupported_crop']}")
    print(f"[Non-Plant Test] Plant diagnosis: {morph['plant_diagnosis']}")
    assert morph['is_unsupported_crop'] is True, "Failed to reject non-plant surface"
    print("PASS: Non-plant input correctly rejected in Stage 1.")

    # 4B: Non-target foliage (synthesize foliage with low cosine similarity to solanaceae)
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)

    # Test non-plant surface through the live endpoint
    res_nonplant = client.post("/api/predict", files={"file": ("table.jpg", buf.tobytes(), "image/jpeg")}).json()
    assert res_nonplant["status"] == "unsupported_crop", f"Expected unsupported_crop, got {res_nonplant['status']}"
    assert "Unsupported" in res_nonplant["prediction"]["common_name"] or "Non-Plant" in res_nonplant["prediction"]["common_name"]
    print("PASS: Non-plant surface flagged as unsupported_crop via /api/predict.")

    # Test synthesized non-target green leaf (e.g. geometric leaf with random non-solanaceae pattern)
    cucurbit_leaf = np.full((400, 400, 3), (240, 240, 240), dtype=np.uint8)
    cv2.circle(cucurbit_leaf, (200, 200), 140, (30, 180, 50), -1)  # Big round leaf (like pumpkin/water lily)
    # Add non-solanaceae concentric patterns
    for r in range(20, 130, 20):
        cv2.circle(cucurbit_leaf, (200, 200), r, (10, 120, 30), 2)
    _, leaf_buf = cv2.imencode(".jpg", cucurbit_leaf)

    res_leaf = client.post("/api/predict", files={"file": ("pumpkin_leaf.jpg", leaf_buf.tobytes(), "image/jpeg")}).json()
    print(f"[Non-Target Foliage Test] Status: {res_leaf['status']} | Identified: {res_leaf['stage1_plant_verification']['identified_plant']}")
    assert res_leaf["status"] == "unsupported_crop", f"Expected unsupported_crop for non-target leaf, got {res_leaf['status']}"
    print("PASS: Non-target foliage flagged as Unsupported Crop via /api/predict.")

if __name__ == "__main__":
    test_user_tomato()
    test_bell_pepper()
    test_potato()
    test_non_target()
    print("\n==============================================")
    print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("==============================================")
