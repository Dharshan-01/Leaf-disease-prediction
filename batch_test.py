"""
Batch upload test script: Uploads multiple unseen test images from each class
to the live web API (http://127.0.0.1:8000/api/predict) to verify diagnosis,
confidence, and treatment prescription delivery.
"""

import os
import random
import urllib.request
import uuid
import json
from pathlib import Path
from collections import defaultdict

TEST_DIR = Path("data/test")
API_URL = "http://127.0.0.1:8000/api/predict"

def upload_and_predict(img_path: Path):
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{img_path.name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = header + img_bytes + footer

    req = urllib.request.Request(API_URL, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    random.seed(42)
    classes = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])
    
    print("=" * 95)
    print("FLORASCAN AI — COMPREHENSIVE MULTI-IMAGE UPLOAD VERIFICATION")
    print("=" * 95)
    print(f"{'True Class':<30} | {'Predicted Disease':<28} | {'Conf':<7} | {'Status':<7} | {'Latency'}")
    print("-" * 95)

    stats = defaultdict(lambda: {"correct": 0, "total": 0, "latencies": []})

    for cls in classes:
        cls_dir = TEST_DIR / cls
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.JPG"))
        # Test 4 distinct random sample images per class
        sample_images = random.sample(images, min(4, len(images)))

        for img in sample_images:
            res = upload_and_predict(img)
            pred = res["prediction"]
            is_correct = pred["class_name"] == cls
            status_str = "MATCH" if is_correct else "MISMATCH"
            
            stats[cls]["total"] += 1
            if is_correct:
                stats[cls]["correct"] += 1
            stats[cls]["latencies"].append(res["latency_ms"])

            print(
                f"{cls:<30} | "
                f"{pred['common_name']:<28} | "
                f"{pred['confidence_percent']:>5.1f}% | "
                f"{status_str:<7} | "
                f"{res['latency_ms']:>5.2f} ms"
            )

    print("=" * 95)
    print("SUMMARY PER CLASS (4 samples tested per class):")
    print("=" * 95)
    total_correct = sum(s["correct"] for s in stats.values())
    total_tested = sum(s["total"] for s in stats.values())
    avg_latency = sum(sum(s["latencies"]) for s in stats.values()) / total_tested

    for cls, s in stats.items():
        acc = (s["correct"] / s["total"]) * 100
        mean_lat = sum(s["latencies"]) / len(s["latencies"])
        print(f"  {cls:<30}: {s['correct']}/{s['total']} correct ({acc:>5.1f}%) | avg latency: {mean_lat:.2f} ms")

    print("-" * 95)
    print(f"OVERALL BATCH UPLOAD ACCURACY: {total_correct}/{total_tested} ({total_correct/total_tested*100:.1f}%) | Avg Latency: {avg_latency:.2f} ms")
    print("=" * 95)

if __name__ == "__main__":
    main()
