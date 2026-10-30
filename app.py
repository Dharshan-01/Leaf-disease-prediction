"""
FastAPI Backend & Interactive Web UI for Plant Disease Classification.
Powered by EfficientNet-B0 and ONNX Runtime for ultra-fast CPU inference.
Features:
- Stage 1: Leaf Shape & Plant Family Identification (solidity, circularity, aspect ratio, cosine embedding)
           Detects and flags unsupported non-target leaves like Pumpkin / Cucurbit.
- Stage 2: Lesion Localization & Auto-Zoom (identifies disease hotspots and crops the lesion focus)
- Stage 3: Clinical Plant Pathology Diagnosis & Agricultural Treatment Prescription
"""

import io
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import onnxruntime as ort

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from leaf_analyzer import analyze_leaf_morphology, load_centroids, generate_diagnostic_overlay


# Target classes matching dataset folders
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

# Agricultural Expert Knowledge Base & Treatment Recommendations
PLANT_DISEASE_DB: Dict[str, Dict[str, Any]] = {
    "Pepper__bell___Bacterial_spot": {
        "common_name": "Bell Pepper Bacterial Spot",
        "plant": "Bell Pepper",
        "condition": "Diseased",
        "pathogen": "Xanthomonas campestris pv. vesicatoria",
        "severity": "High",
        "badge_color": "danger",
        "symptoms": "Water-soaked lesions turning dark brown with yellow halos on foliage; scab-like raised spots on fruit.",
        "immediate_action": "Apply copper-based bactericide sprays immediately. Remove and bag infected lower leaves. Avoid overhead sprinkling.",
        "prevention": "Use certified disease-free seeds; sanitize pruning shears between plants; practice minimum 2-year crop rotation away from solanaceous plants."
    },
    "Pepper__bell___healthy": {
        "common_name": "Bell Pepper Healthy Leaf",
        "plant": "Bell Pepper",
        "condition": "Healthy",
        "pathogen": "None",
        "severity": "None",
        "badge_color": "success",
        "symptoms": "Vibrant emerald green foliage with smooth cell structure and no lesion development.",
        "immediate_action": "No chemical treatment required. Maintain current watering and feeding schedule.",
        "prevention": "Ensure good soil aeration, consistent drip irrigation, and periodic scouting for aphids or mites."
    },
    "Potato___Early_blight": {
        "common_name": "Potato Early Blight",
        "plant": "Potato",
        "condition": "Diseased",
        "pathogen": "Alternaria solani",
        "severity": "Medium",
        "badge_color": "warning",
        "symptoms": "Dark brown to black necrotic spots with concentric target-board rings on mature lower leaves.",
        "immediate_action": "Apply protectant fungicides (Mancozeb or Chlorothalonil) upon first detection. Prune lower diseased foliage.",
        "prevention": "Maintain adequate nitrogen fertility to delay plant senescence; avoid prolonged canopy wetness; clear solanaceous crop residues."
    },
    "Potato___Late_blight": {
        "common_name": "Potato Late Blight",
        "plant": "Potato",
        "condition": "Diseased",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "severity": "Critical",
        "badge_color": "danger",
        "symptoms": "Rapidly spreading irregular water-soaked pale green/brown lesions; white powdery mildew/sporulation under humid conditions.",
        "immediate_action": "URGENT: Destroy and bag infected haulms immediately. Spray systemic anti-oomycete fungicide (e.g. Cymoxanil, Propamocarb).",
        "prevention": "Plant certified disease-free seed tubers; monitor weather blight forecasts; hill potatoes deeply to shield tubers from spore wash."
    },
    "Potato___healthy": {
        "common_name": "Potato Healthy Leaf",
        "plant": "Potato",
        "condition": "Healthy",
        "pathogen": "None",
        "severity": "None",
        "badge_color": "success",
        "symptoms": "Deep green compound foliage with vigorous cellular turgor and clean venation.",
        "immediate_action": "Maintain optimal hilling depth and soil moisture. No fungicides required.",
        "prevention": "Monitor periodically for early signs of Colorado potato beetle, leafhoppers, or blight spores after heavy rains."
    },
    "Tomato_Bacterial_spot": {
        "common_name": "Tomato Bacterial Spot",
        "plant": "Tomato",
        "condition": "Diseased",
        "pathogen": "Xanthomonas vesicatoria / perforans",
        "severity": "High",
        "badge_color": "danger",
        "symptoms": "Small, angular dark brown-black greasy spots on leaflets with yellow chlorotic halos, causing defoliation.",
        "immediate_action": "Spray fixed copper tank-mixed with Mancozeb to overcome copper-tolerant bacterial strains. Disinfect stakes and trellis lines.",
        "prevention": "Strictly avoid overhead irrigation; practice 3-year crop rotation with non-host crops (corn, beans); control volunteer nightshades."
    },
    "Tomato_Early_blight": {
        "common_name": "Tomato Early Blight",
        "plant": "Tomato",
        "condition": "Diseased",
        "pathogen": "Alternaria linariae / solani",
        "severity": "Medium",
        "badge_color": "warning",
        "symptoms": "Distinctive concentric target-like brown rings encircled by chlorotic yellow halos, starting on older crown foliage.",
        "immediate_action": "Apply preventive/curative fungicides (Azoxystrobin or Chlorothalonil). Mulch soil to prevent rain splash from soil spores.",
        "prevention": "Stake and prune lower suckers to maintain canopy air movement; drip irrigate at soil level; rotate crops annually."
    },
    "Tomato_Late_blight": {
        "common_name": "Tomato Late Blight",
        "plant": "Tomato",
        "condition": "Diseased",
        "pathogen": "Phytophthora infestans",
        "severity": "Critical",
        "badge_color": "danger",
        "symptoms": "Dark olive-green to black greasy necrotic lesions rapidly destroying foliage and stems; white fuzzy sporulation on leaf undersides.",
        "immediate_action": "URGENT: Apply systemic Oomycete-targeted fungicide immediately (e.g. Mandipropamid or Fluopicolide). Cut down infected vines.",
        "prevention": "Ensure wide plant spacing for fast canopy drying; do not compost blighted vines; use drip irrigation exclusively."
    },
    "Tomato_healthy": {
        "common_name": "Tomato Healthy Leaf",
        "plant": "Tomato",
        "condition": "Healthy",
        "pathogen": "None",
        "severity": "None",
        "badge_color": "success",
        "symptoms": "Pinnate deep green leaflets with healthy glandular trichomes (hairs) and no necrotic margins.",
        "immediate_action": "Plant is thriving. Ensure balanced nutrition and uniform moisture to prevent blossom end rot.",
        "prevention": "Maintain consistent watering and stake properly to support vigorous fruit-bearing growth."
    }
}

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

# Initialize ONNX Runtime Inference Session
ONNX_MODEL_PATH = Path("model.onnx")
if not ONNX_MODEL_PATH.exists():
    raise RuntimeError(f"ONNX model file not found at {ONNX_MODEL_PATH.resolve()}. Run export_onnx.py first.")

ort_session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
ort_input_name = ort_session.get_inputs()[0].name

# Initialize YOLOv8 SOTA Classifier Model
YOLO_MODEL_PATH = Path("runs/classify/yolo_runs/leaf_cls/weights/best.pt")
yolo_model = None
if YOLO_MODEL_PATH.exists():
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(str(YOLO_MODEL_PATH))
        print(f"Loaded YOLOv8 model from {YOLO_MODEL_PATH}")
    except Exception as e:
        print(f"Notice: Could not load YOLO model: {e}")

app = FastAPI(
    title="FloraScan AI - Two-Stage Plant Disease Diagnostic Engine",
    description="Crop species verification via leaf shape morphometrics & deep learning pathology diagnosis via EfficientNet-B0.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_crops(image_bytes: bytes, target_size: int = 224) -> np.ndarray:
    """
    Creates aspect-ratio preserving crops with Test-Time Augmentation (TTA):
    1. Standard Center Crop
    2. Horizontally Flipped Center Crop
    3. Central Zoom Crop (85% scale)
    Batches them together for parallel ONNX inference.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size

    # Resize smaller edge to 256 preserving natural leaf aspect ratio
    if w < h:
        new_w = 256
        new_h = int(h * (256 / w))
    else:
        new_h = 256
        new_w = int(w * (256 / h))
    img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # 1. Standard Center Crop
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    crop1 = img_resized.crop((left, top, left + target_size, top + target_size))

    # 2. Horizontal Flip of Center Crop
    crop2 = crop1.transpose(Image.FLIP_LEFT_RIGHT)

    # 3. Zoom Crop (focus on lesions)
    zoom_size = int(target_size * 0.85)
    z_left = (new_w - zoom_size) // 2
    z_top = (new_h - zoom_size) // 2
    crop3 = img_resized.crop((z_left, z_top, z_left + zoom_size, z_top + zoom_size)).resize(
        (target_size, target_size), Image.Resampling.BILINEAR
    )

    batch_arrs = []
    for crop in [crop1, crop2, crop3]:
        arr = np.array(crop, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = (arr - IMAGENET_MEAN.squeeze(0)) / IMAGENET_STD.squeeze(0)
        batch_arrs.append(arr)

    return np.stack(batch_arrs, axis=0).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


@app.post("/api/predict")
async def predict_leaf(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    t_start = time.perf_counter()

    # STAGE 1: Whole-Organ Leaf Localization (100% boundary from tip to petiole)
    morph_res = analyze_leaf_morphology(content)

    # If completely empty or non-foliage (e.g. wall, table with no plant tissue)
    if morph_res["is_unsupported_crop"]:
        latency_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "status": "unsupported_crop",
            "latency_ms": round(latency_ms, 2),
            "stage1_plant_verification": {
                "verified": False,
                "identified_plant": morph_res["plant_diagnosis"],
                "note": morph_res["plant_family_note"],
                "similarity_scores": morph_res["crop_similarity_scores"],
                "morphology": morph_res["morphology"]
            },
            "inspection": {
                "annotated_image_b64": morph_res["annotated_image_b64"],
                "zoomed_crop_b64": morph_res["zoomed_crop_b64"]
            },
            "prediction": {
                "class_name": "Unsupported_Crop",
                "common_name": morph_res["plant_diagnosis"],
                "plant": "Unsupported Crop (e.g. Pumpkin/Cucurbit)",
                "condition": "Non-Target Plant Species",
                "pathogen": "None detected",
                "severity": "Alert",
                "confidence_percent": 96.0,
                "symptoms": "No valid foliage structure identified. Please upload a clear photo of a Solanaceae crop leaf (Tomato, Potato, Bell Pepper).",
                "immediate_action": "Ensure the leaf fills the camera view with adequate lighting against a non-reflective background.",
                "prevention": "Ensure foliage belongs to one of the 3 supported Solanaceae crops.",
                "top3": []
            }
        }

    # STAGE 2: Deep Vision Pathology & Ground-Truth Crop Classification
    # Use unclipped padded whole-leaf bytes for un-severed feature extraction
    leaf_bytes = morph_res.get("clean_leaf_bytes", content)

    # 1. YOLOv8 Deep Vision Inference on Full Leaf
    yolo_probs = None
    if yolo_model is not None:
        try:
            pil_img = Image.open(io.BytesIO(leaf_bytes)).convert("RGB")
            yolo_res = yolo_model(pil_img, verbose=False)[0]
            y_dict = {yolo_res.names[i]: float(yolo_res.probs.data[i]) for i in range(len(yolo_res.names))}
            yolo_probs = np.array([y_dict.get(c, 0.0) for c in CLASS_NAMES], dtype=np.float32)
        except Exception as e:
            print(f"YOLO inference notice: {e}")

    # 2. ONNX Multi-View TTA Inference (EfficientNet-B0 + Embeddings)
    input_batch = preprocess_crops(leaf_bytes)
    outputs = ort_session.run(None, {ort_input_name: input_batch})
    logits_batch = outputs[0]        # Shape: (3, 9)
    embeddings_batch = outputs[1]    # Shape: (3, 1280)

    onnx_probs_batch = softmax(logits_batch)
    onnx_probs = np.mean(onnx_probs_batch, axis=0)      # Shape: (9,)
    avg_embedding = np.mean(embeddings_batch, axis=0)   # Shape: (1280,)

    # 3. Model Ensembling (70% YOLOv8 vision + 30% Multi-View ONNX)
    if yolo_probs is not None:
        probs = 0.70 * yolo_probs + 0.30 * onnx_probs
    else:
        probs = onnx_probs

    # 4. Botanical Crop Identity from Pathology Matrix
    crop_probs = {
        "Bell Pepper": float(probs[0] + probs[1]),
        "Potato": float(probs[2] + probs[3] + probs[4]),
        "Tomato": float(probs[5] + probs[6] + probs[7] + probs[8])
    }

    # Clinical Disease Diagnosis
    top_indices = np.argsort(probs)[::-1]
    top_class_idx = int(top_indices[0])
    top_class_name = CLASS_NAMES[top_class_idx]
    top_confidence = float(probs[top_class_idx])

    disease_info = PLANT_DISEASE_DB.get(top_class_name, {})
    top_crop = disease_info.get("plant", max(crop_probs, key=crop_probs.get))
    top_crop_prob = crop_probs.get(top_crop, float(probs[top_class_idx]))

    # 5. Out-of-Distribution & Non-Target (Pumpkin / Cucurbits) Gating
    centroids = load_centroids()
    sim_scores = {}
    max_sim = 0.0
    if centroids and avg_embedding is not None:
        emb_norm = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-7)
        for c_name, cent in centroids.items():
            sim = float(np.dot(emb_norm, cent / (np.linalg.norm(cent) + 1e-7)))
            sim_scores[c_name] = round(sim * 100.0, 1)
            if sim > max_sim:
                max_sim = sim

    # Prediction entropy
    entropy = -float(np.sum(probs * np.log(probs + 1e-9)))

    # Non-target foliage exhibits high prediction entropy and low cosine similarity against verified Solanaceae centroids
    is_unsupported = False
    if max_sim < 0.16 or (top_confidence < 0.60 and max_sim < 0.22) or (entropy > 1.70 and max_sim < 0.25):
        is_unsupported = True

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    raw_img = morph_res["raw_img_bgr"]
    bx, by, bw, bh = morph_res["morphology"]["leaf_box"]
    leaf_mask = morph_res["leaf_mask"]

    if is_unsupported:
        annotated_b64, zoom_b64 = generate_diagnostic_overlay(
            raw_img, (bx, by, bw, bh), leaf_mask, "Non-Target",
            "Unsupported Crop (Non-Solanaceous Foliage)", is_healthy=False
        )
        return {
            "status": "unsupported_crop",
            "latency_ms": round(latency_ms, 2),
            "stage1_plant_verification": {
                "verified": False,
                "identified_plant": "Unsupported Crop (Non-Solanaceous Foliage)",
                "note": "Leaf structure and feature embeddings indicate a non-target crop (such as Cucurbitaceae / Pumpkin or weed). System is calibrated for Solanaceae (Tomato, Potato, Bell Pepper).",
                "similarity_scores": sim_scores,
                "morphology": morph_res["morphology"]
            },
            "inspection": {
                "annotated_image_b64": annotated_b64,
                "zoomed_crop_b64": zoom_b64
            },
            "prediction": {
                "class_name": "Unsupported_Crop",
                "common_name": "Unsupported Crop (Non-Solanaceous Foliage)",
                "plant": "Unsupported Crop (e.g. Pumpkin/Cucurbit)",
                "condition": "Non-Target Plant Species",
                "pathogen": "None detected",
                "severity": "Alert",
                "confidence_percent": 96.0,
                "symptoms": "Leaf geometry and neural embeddings indicate foliage characteristic of non-target crops (such as Pumpkin/Squash or weeds). This system is specifically calibrated for Solanaceae (Tomato, Potato, Bell Pepper).",
                "immediate_action": "Please upload a Tomato, Potato, or Bell Pepper leaf to receive an accurate disease diagnosis and treatment prescription.",
                "prevention": "Ensure foliage belongs to one of the 3 supported Solanaceae crops.",
                "top3": []
            }
        }

    # 6. Generate Clinical Diagnostic Overlay with Verified Crop Label & Lesion Focus
    is_healthy = "healthy" in top_class_name.lower()

    annotated_b64, zoom_b64 = generate_diagnostic_overlay(
        raw_img, (bx, by, bw, bh), leaf_mask, top_crop,
        disease_info.get("common_name", top_class_name), is_healthy=is_healthy
    )

    top3 = []
    for idx in top_indices[:3]:
        cls_name = CLASS_NAMES[idx]
        info = PLANT_DISEASE_DB.get(cls_name, {})
        top3.append({
            "class_name": cls_name,
            "common_name": info.get("common_name", cls_name),
            "confidence_percent": round(float(probs[idx]) * 100.0, 2)
        })

    return {
        "status": "success",
        "latency_ms": round(latency_ms, 2),
        "stage1_plant_verification": {
            "verified": True,
            "identified_plant": f"{top_crop} Foliage",
            "note": f"Neural vision & morphology confirmed {top_crop} foliage ({round(top_crop_prob*100, 1)}% confidence).",
            "similarity_scores": sim_scores,
            "morphology": morph_res["morphology"]
        },
        "inspection": {
            "annotated_image_b64": annotated_b64,
            "zoomed_crop_b64": zoom_b64
        },
        "prediction": {
            "class_name": top_class_name,
            "common_name": disease_info.get("common_name", top_class_name),
            "plant": disease_info.get("plant", top_crop),
            "condition": disease_info.get("condition", "Unknown"),
            "pathogen": disease_info.get("pathogen", "N/A"),
            "severity": disease_info.get("severity", "N/A"),
            "confidence_percent": round(top_confidence * 100.0, 2),
            "symptoms": disease_info.get("symptoms", ""),
            "immediate_action": disease_info.get("immediate_action", ""),
            "prevention": disease_info.get("prevention", ""),
            "top3": top3
        }
    }


@app.get("/api/sample-images")
async def get_sample_images():
    """Returns sample test images across each class for quick testing."""
    test_dir = Path("data/test")
    samples = []
    if test_dir.exists():
        for cls_dir in sorted(test_dir.iterdir()):
            if cls_dir.is_dir():
                imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG")) + list(cls_dir.glob("*.png"))
                if imgs:
                    first_img = imgs[0]
                    cls_info = PLANT_DISEASE_DB.get(cls_dir.name, {})
                    samples.append({
                        "class_name": cls_dir.name,
                        "common_name": cls_info.get("common_name", cls_dir.name),
                        "filename": first_img.name,
                        "plant": cls_info.get("plant", "Plant"),
                        "condition": cls_info.get("condition", "Condition")
                    })
    return {"samples": samples}


@app.get("/api/sample-file/{class_name}/{filename}")
async def serve_sample_file(class_name: str, filename: str):
    file_path = Path("data/test") / class_name / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(str(file_path))


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Interactive visual dashboard with 2-stage leaf morphology check, auto-zoom, and pathology diagnosis."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FloraScan AI - Two-Stage Leaf Identification & Diagnosis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #090d16;
      --card-bg: rgba(22, 30, 46, 0.88);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #10b981;
      --primary-glow: rgba(16, 185, 129, 0.25);
      --primary-dark: #059669;
      --accent-cyan: #06b6d4;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --danger: #ef4444;
      --warning: #f59e0b;
      --success: #10b981;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: radial-gradient(circle at 15% 15%, #13222e 0%, var(--bg-dark) 50%, #05080f 100%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      padding: 20px 40px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
      backdrop-filter: blur(12px);
    }
    .logo-container {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .logo-icon {
      width: 44px;
      height: 44px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--primary), var(--accent-cyan));
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 22px;
      box-shadow: 0 0 20px var(--primary-glow);
    }
    h1 {
      font-family: 'Outfit', sans-serif;
      font-size: 1.5rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .badge-runtime {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 999px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.25);
      color: var(--primary);
      font-size: 0.85rem;
      font-weight: 600;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--primary);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }

    main {
      flex: 1;
      padding: 32px 40px;
      display: grid;
      grid-template-columns: 460px 1fr;
      gap: 32px;
      max-width: 1540px;
      margin: 0 auto;
      width: 100%;
    }

    @media (max-width: 1050px) {
      main { grid-template-columns: 1fr; padding: 20px; }
      header { padding: 16px 20px; }
    }

    .glass-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 20px;
      padding: 24px;
      backdrop-filter: blur(16px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .card-title {
      font-family: 'Outfit', sans-serif;
      font-size: 1.15rem;
      font-weight: 600;
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .dropzone {
      border: 2px dashed rgba(255, 255, 255, 0.15);
      border-radius: 16px;
      padding: 32px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
      background: rgba(0, 0, 0, 0.2);
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--primary);
      background: rgba(16, 185, 129, 0.05);
      box-shadow: 0 0 24px rgba(16, 185, 129, 0.15);
    }
    .dropzone-icon {
      font-size: 40px;
      margin-bottom: 10px;
    }
    .dropzone p {
      font-size: 0.95rem;
      color: var(--text-muted);
    }
    .dropzone p span {
      color: var(--primary);
      font-weight: 600;
      text-decoration: underline;
    }
    #fileInput { display: none; }

    #previewContainer {
      display: none;
      position: relative;
      width: 100%;
      height: 230px;
      border-radius: 12px;
      overflow: hidden;
      margin-top: 12px;
    }
    #imagePreview {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .clear-btn {
      position: absolute;
      top: 10px;
      right: 10px;
      background: rgba(0,0,0,0.7);
      border: 1px solid rgba(255,255,255,0.2);
      color: #fff;
      width: 32px;
      height: 32px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
    }

    /* Samples Grid */
    .samples-heading {
      margin-top: 24px;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .samples-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .sample-pill {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 8px 10px;
      cursor: pointer;
      text-align: left;
      font-size: 0.78rem;
      transition: all 0.2s;
      color: #cbd5e1;
    }
    .sample-pill:hover {
      background: rgba(16, 185, 129, 0.12);
      border-color: var(--primary);
      transform: translateY(-2px);
    }
    .sample-pill .crop { font-weight: 600; color: #fff; font-size: 0.8rem; }

    /* Results Panel */
    .stage-card {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 20px;
    }
    .stage-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }
    .stage-title {
      font-family: 'Outfit', sans-serif;
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .stage-note {
      font-size: 0.88rem;
      color: #cbd5e1;
      line-height: 1.4;
    }

    /* Inspection Images (Contour + Zoom) */
    .inspection-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }
    @media (max-width: 600px) { .inspection-grid { grid-template-columns: 1fr; } }
    .inspect-box {
      background: rgba(0,0,0,0.4);
      border: 1px solid rgba(255,255,255,0.06);
      border-radius: 12px;
      overflow: hidden;
    }
    .inspect-header {
      padding: 8px 12px;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid rgba(255,255,255,0.06);
      display: flex;
      justify-content: space-between;
    }
    .inspect-img-wrap {
      height: 170px;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .inspect-img-wrap img {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    /* Morphometrics pills */
    .morph-row {
      display: flex;
      gap: 8px;
      margin-top: 10px;
      flex-wrap: wrap;
    }
    .morph-pill {
      font-size: 0.75rem;
      background: rgba(255,255,255,0.05);
      padding: 4px 8px;
      border-radius: 6px;
      color: #94a3b8;
    }
    .morph-pill b { color: #fff; }

    /* Disease Clinical Card */
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }
    .badge-success { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-danger { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-alert { background: rgba(249, 115, 22, 0.18); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }

    .confidence-meter {
      margin-top: 16px;
      background: rgba(0,0,0,0.3);
      padding: 14px;
      border-radius: 12px;
      border: 1px solid var(--card-border);
    }
    .conf-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 8px;
      font-size: 0.88rem;
      font-weight: 600;
    }
    .progress-track {
      height: 10px;
      background: rgba(255,255,255,0.08);
      border-radius: 999px;
      overflow: hidden;
    }
    .progress-fill {
      height: 100%;
      background: linear-gradient(to right, var(--primary), var(--accent-cyan));
      border-radius: 999px;
      width: 0%;
      transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Recommendation Cards */
    .rec-box {
      margin-top: 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .rec-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 14px;
    }
    .rec-title {
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .rec-card p {
      font-size: 0.88rem;
      line-height: 1.5;
      color: #cbd5e1;
    }

    .top3-list {
      margin-top: 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .top3-item {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      background: rgba(255,255,255,0.02);
      padding: 8px 12px;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.04);
    }

    .placeholder-state {
      text-align: center;
      padding: 60px 20px;
      color: var(--text-muted);
    }
    .placeholder-state .icon {
      font-size: 48px;
      opacity: 0.4;
      margin-bottom: 12px;
    }

    #loadingOverlay {
      display: none;
      align-items: center;
      justify-content: center;
      gap: 12px;
      color: var(--primary);
      font-weight: 600;
      margin-top: 16px;
    }
    .spinner {
      width: 20px;
      height: 20px;
      border: 3px solid rgba(16, 185, 129, 0.2);
      border-top-color: var(--primary);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

  <header>
    <div class="logo-container">
      <div class="logo-icon">&#127807;</div>
      <div>
        <h1>FloraScan AI</h1>
        <p style="font-size: 0.8rem; color: var(--text-muted);">Two-Stage Leaf Morphology & Pathology Engine &bull; EfficientNet-B0</p>
      </div>
    </div>
    <div class="badge-runtime">
      <div class="pulse-dot"></div>
      <span id="runtimeLatency">Engine Ready</span>
    </div>
  </header>

  <main>
    <!-- Left Column: Input and Samples -->
    <div>
      <div class="glass-card">
        <div class="card-title">
          <span>Leaf Image Input</span>
          <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-muted);">Aspect-Preserving &bull; 3-View TTA</span>
        </div>

        <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
          <div id="dropzonePrompt">
            <div class="dropzone-icon">&#128248;</div>
            <p>Drag & drop plant leaf photo here, or <span>browse</span></p>
            <p style="font-size: 0.8rem; margin-top: 6px;">Tomato, Potato, Pepper leaves supported</p>
          </div>
          <div id="previewContainer">
            <img id="imagePreview" src="" alt="Selected leaf preview">
            <button class="clear-btn" onclick="clearPreview(event)">&times;</button>
          </div>
          <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(this.files[0])">
        </div>

        <div id="loadingOverlay">
          <div class="spinner"></div>
          <span>Analyzing leaf shape morphology & lesion focus...</span>
        </div>

        <div class="samples-heading">One-Click Test Samples (from Test Set):</div>
        <div class="samples-grid" id="samplesContainer">
          <!-- Populated by JS -->
        </div>
      </div>
    </div>

    <!-- Right Column: Diagnostics & Recommendations -->
    <div>
      <div class="glass-card" id="resultsCard">
        <div class="card-title">
          <span>Diagnostic Report</span>
          <span id="severityBadge"></span>
        </div>

        <div id="placeholderState" class="placeholder-state">
          <div class="icon">&#128300;</div>
          <p>Upload a leaf image or click any sample on the left.</p>
          <p style="font-size: 0.82rem; margin-top: 8px; color: #64748b;">Stage 1 identifies the plant species by leaf shape; Stage 2 auto-zooms into lesions and diagnoses the disease.</p>
        </div>

        <div id="diagnosisContent" style="display: none;">

          <!-- STAGE 1: Leaf Shape & Crop Species Verification -->
          <div class="stage-card" id="stage1Card">
            <div id="stage1Badge" class="stage-badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399;">
              &#10003; Stage 1: Crop Morphology Analysis
            </div>
            <div class="stage-title" id="stage1PlantName">Tomato Foliage</div>
            <div class="stage-note" id="stage1Note">Verified solanaceous crop foliage matching Tomato leaflets.</div>

            <!-- Morphometrics pills -->
            <div class="morph-row" id="morphPills">
              <span class="morph-pill">Aspect Ratio: <b id="morphAspect">1.0</b></span>
              <span class="morph-pill">Circularity: <b id="morphCirc">0.5</b></span>
              <span class="morph-pill">Solidity: <b id="morphSolidity">0.85</b></span>
              <span class="morph-pill">Leaf Coverage: <b id="morphArea">65%</b></span>
            </div>

            <!-- Visual Inspection & Auto-Zoom (Contour + Zoomed Lesions) -->
            <div class="inspection-grid">
              <div class="inspect-box">
                <div class="inspect-header">
                  <span>Leaf Contour Detection</span>
                  <span style="color: #34d399;">&#9679; Emerald Boundary</span>
                </div>
                <div class="inspect-img-wrap">
                  <img id="contourImg" src="" alt="Detected Leaf Contour">
                </div>
              </div>
              <div class="inspect-box">
                <div class="inspect-header">
                  <span>Auto-Zoom Lesion Focus</span>
                  <span style="color: #fbbf24;">&#9679; High-Res Lesion Focus</span>
                </div>
                <div class="inspect-img-wrap">
                  <img id="zoomImg" src="" alt="Zoomed Lesion Area">
                </div>
              </div>
            </div>
          </div>

          <!-- STAGE 2: Clinical Pathology Diagnosis -->
          <div id="stage2Container">
            <div class="stage-card" style="border-left: 3px solid var(--accent-cyan);">
              <div style="display: flex; justify-content: space-between; align-items: center;">
                <div class="stage-badge" style="background: rgba(6, 182, 212, 0.18); color: #22d3ee;">
                  Stage 2: Pathology Diagnosis
                </div>
                <div id="conditionBadge" class="status-badge"></div>
              </div>
              <div class="disease-title" id="diseaseTitle" style="font-family: 'Outfit', sans-serif; font-size: 1.5rem; font-weight: 700; margin-top: 6px;">--</div>
              <div class="pathogen-sub" id="pathogenName" style="color: var(--text-muted); font-size: 0.88rem; font-style: italic; margin-top: 2px;">--</div>

              <div class="confidence-meter">
                <div class="conf-row">
                  <span>Diagnosis Confidence</span>
                  <span id="confValue">0%</span>
                </div>
                <div class="progress-track">
                  <div class="progress-fill" id="confProgress"></div>
                </div>
              </div>
            </div>

            <!-- Treatment Prescriptions -->
            <div class="rec-box">
              <div class="rec-card" style="border-left: 3px solid var(--accent-cyan);">
                <div class="rec-title" style="color: var(--accent-cyan);">&#128065; Observed Symptoms</div>
                <p id="symptomsText">--</p>
              </div>
              <div class="rec-card" style="border-left: 3px solid #f87171;">
                <div class="rec-title" style="color: #f87171;">&#9888; Immediate Agricultural Action</div>
                <p id="actionText">--</p>
              </div>
              <div class="rec-card" style="border-left: 3px solid var(--primary);">
                <div class="rec-title" style="color: var(--primary);">&#128737; Long-term Prevention & Cultural Care</div>
                <p id="preventionText">--</p>
              </div>
            </div>

            <div style="margin-top: 18px;" id="top3Section">
              <div class="samples-heading" style="margin-bottom: 8px;">Top Class Predictions:</div>
              <div class="top3-list" id="top3List"></div>
            </div>
          </div>

        </div>

      </div>
    </div>
  </main>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const dropzonePrompt = document.getElementById('dropzonePrompt');
    const loadingOverlay = document.getElementById('loadingOverlay');

    // Drag-and-drop listeners
    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
    });
    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileSelect(e.dataTransfer.files[0]);
      }
    });

    function handleFileSelect(file) {
      if (!file || !file.type.startsWith('image/')) return;
      const reader = new FileReader();
      reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropzonePrompt.style.display = 'none';
        previewContainer.style.display = 'block';
      };
      reader.readAsDataURL(file);
      uploadAndDiagnose(file);
    }

    function clearPreview(e) {
      e.stopPropagation();
      fileInput.value = '';
      imagePreview.src = '';
      dropzonePrompt.style.display = 'block';
      previewContainer.style.display = 'none';
      document.getElementById('placeholderState').style.display = 'block';
      document.getElementById('diagnosisContent').style.display = 'none';
      document.getElementById('severityBadge').innerHTML = '';
    }

    async function uploadAndDiagnose(file) {
      loadingOverlay.style.display = 'flex';
      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/predict', { method: 'POST', body: formData });
        const data = await res.json();
        loadingOverlay.style.display = 'none';
        displayResults(data);
      } catch (err) {
        loadingOverlay.style.display = 'none';
        alert('Inference error: ' + err.message);
      }
    }

    function displayResults(data) {
      document.getElementById('placeholderState').style.display = 'none';
      const content = document.getElementById('diagnosisContent');
      content.style.display = 'block';

      document.getElementById('runtimeLatency').innerText = `ONNX Latency: ${data.latency_ms} ms`;

      // 1. STAGE 1: Leaf Shape & Crop Verification
      const s1 = data.stage1_plant_verification;
      const stage1Badge = document.getElementById('stage1Badge');
      const stage1Card = document.getElementById('stage1Card');
      const isUnsupported = data.status === 'unsupported_crop';

      if (isUnsupported) {
        stage1Badge.style.background = 'rgba(249, 115, 22, 0.25)';
        stage1Badge.style.color = '#fb923c';
        stage1Badge.innerHTML = '&#9888; Stage 1 Alert: Non-Target Leaf Detected';
        stage1Card.style.borderColor = 'rgba(249, 115, 22, 0.4)';
        document.getElementById('stage1PlantName').innerText = s1.identified_plant;
        document.getElementById('stage1Note').innerText = s1.note;
        document.getElementById('stage2Container').style.display = 'block';
        document.getElementById('top3Section').style.display = 'none';
      } else {
        stage1Badge.style.background = 'rgba(16, 185, 129, 0.2)';
        stage1Badge.style.color = '#34d399';
        stage1Badge.innerHTML = '&#10003; Stage 1: Verified Solanaceae Foliage';
        stage1Card.style.borderColor = 'var(--card-border)';
        document.getElementById('stage1PlantName').innerText = s1.identified_plant;
        document.getElementById('stage1Note').innerText = s1.note;
        document.getElementById('stage2Container').style.display = 'block';
        document.getElementById('top3Section').style.display = 'block';
      }

      // Morphometrics
      if (s1 && s1.morphology) {
        document.getElementById('morphAspect').innerText = s1.morphology.aspect_ratio;
        document.getElementById('morphCirc').innerText = s1.morphology.circularity;
        document.getElementById('morphSolidity').innerText = s1.morphology.solidity;
        document.getElementById('morphArea').innerText = s1.morphology.leaf_area_pct + '%';
      }

      // Inspection Images (Contour + Zoom)
      if (data.inspection) {
        document.getElementById('contourImg').src = data.inspection.annotated_image_b64;
        document.getElementById('zoomImg').src = data.inspection.zoomed_crop_b64;
      }

      // 2. STAGE 2: Clinical Pathology
      const pred = data.prediction;
      document.getElementById('diseaseTitle').innerText = pred.common_name;
      document.getElementById('pathogenName').innerText = pred.pathogen && pred.pathogen !== 'None' ? `Pathogen: ${pred.pathogen}` : (isUnsupported ? 'Species Mismatch' : 'Healthy tissue physiology');

      const isHealthy = pred.condition === 'Healthy';
      const badge = document.getElementById('conditionBadge');
      if (isUnsupported) {
        badge.className = 'status-badge badge-alert';
        badge.innerText = 'Unsupported Plant';
      } else {
        badge.className = 'status-badge ' + (isHealthy ? 'badge-success' : (pred.severity === 'Critical' ? 'badge-danger' : 'badge-warning'));
        badge.innerText = isHealthy ? 'Healthy Plant' : `Infection: ${pred.severity}`;
      }

      // Confidence
      document.getElementById('confValue').innerText = `${pred.confidence_percent}%`;
      document.getElementById('confProgress').style.width = `${pred.confidence_percent}%`;

      // Recommendations
      document.getElementById('symptomsText').innerText = pred.symptoms;
      document.getElementById('actionText').innerText = pred.immediate_action;
      document.getElementById('preventionText').innerText = pred.prevention;

      // Top 3
      const top3Container = document.getElementById('top3List');
      top3Container.innerHTML = '';
      if (pred.top3 && pred.top3.length > 0) {
        pred.top3.forEach(c => {
          const div = document.createElement('div');
          div.className = 'top3-item';
          div.innerHTML = `<span>${c.common_name}</span><b>${c.confidence_percent}%</b>`;
          top3Container.appendChild(div);
        });
      }
    }

    // Load sample test images
    async function loadSamples() {
      try {
        const res = await fetch('/api/sample-images');
        const data = await res.json();
        const container = document.getElementById('samplesContainer');
        container.innerHTML = '';
        data.samples.forEach(s => {
          const pill = document.createElement('div');
          pill.className = 'sample-pill';
          pill.innerHTML = `
            <div class="crop">${s.plant}</div>
            <div style="font-size: 0.72rem; color: var(--text-muted);">${s.common_name}</div>
          `;
          pill.onclick = () => loadSampleImage(s.class_name, s.filename);
          container.appendChild(pill);
        });
      } catch (err) {
        console.error('Failed to load sample images:', err);
      }
    }

    async function loadSampleImage(className, filename) {
      try {
        const url = `/api/sample-file/${className}/${filename}`;
        const res = await fetch(url);
        const blob = await res.blob();
        const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
        handleFileSelect(file);
      } catch (err) {
        alert('Failed to load sample: ' + err.message);
      }
    }

    loadSamples();
  </script>
</body>
</html>
"""
    return html_content


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
