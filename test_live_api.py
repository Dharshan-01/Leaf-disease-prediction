import cv2
import numpy as np

test_cases = [
    ('Pepper Healthy', 'data/test/Pepper__bell___healthy/002f87b7-e1a5-49e5-a422-bb423630ded5___JR_HL 8068.JPG'),
    ('Pepper Bact Spot', 'data/test/Pepper__bell___Bacterial_spot/01940b6d-7dea-4889-a7b8-a35f4e9bba34___NREC_B.Spot 9120.JPG'),
    ('Tomato Healthy', 'data/test/Tomato_healthy/02b4afdf-e1de-4c0e-a38d-3f19afeb9ea9___RS_HL 0493.JPG'),
    ('Tomato Early Blight', 'data/test/Tomato_Early_blight/07324d80-17c5-4541-9702-62e9c744ac07___RS_Erly.B 7455.JPG'),
    ('Potato Early Blight', 'data/test/Potato___Early_blight/02578b86-b234-4ac0-9bc3-691b5610e2bf___RS_Early.B 7562.JPG'),
    ('User Tomato Leaf', 'test_user_tomato.png')
]

# Use TestClient directly for robust self-contained testing
from fastapi.testclient import TestClient
from app import app
client = TestClient(app)

print("=== TESTING FLUID API (YOLOv8 + WHOLE-ORGAN MORPHOLOGY) ===")
for name, path in test_cases:
    with open(path, 'rb') as f:
        res = client.post('/api/predict', files={'file': ('leaf.jpg', f.read(), 'image/jpeg')}).json()
    status = res.get('status')
    pred = res.get('prediction', {}).get('common_name', 'N/A')
    conf = res.get('prediction', {}).get('confidence_percent', 0.0)
    plant = res.get('stage1_plant_verification', {}).get('identified_plant', 'N/A')
    lat = res.get('latency_ms', 0)
    print(f"[{name}]: Status={status} | Pred={pred} ({conf}%) | Plant={plant} | Latency={lat}ms")

# Test Leaf on Wood Table
leaf = cv2.imread('data/test/Tomato_healthy/02b4afdf-e1de-4c0e-a38d-3f19afeb9ea9___RS_HL 0493.JPG')
lh, lw = leaf.shape[:2]
canvas = np.full((500, 500, 3), [35, 75, 140], dtype=np.uint8)
canvas[120:120+lh, 120:120+lw] = leaf
_, buf = cv2.imencode('.jpg', canvas)
res_table = client.post('/api/predict', files={'file': ('table.jpg', buf.tobytes(), 'image/jpeg')}).json()
t_status = res_table.get('status')
t_pred = res_table.get('prediction', {}).get('common_name')
t_conf = res_table.get('prediction', {}).get('confidence_percent')
t_plant = res_table.get('stage1_plant_verification', {}).get('identified_plant')
print(f"[Leaf on Wood Table]: Status={t_status} | Pred={t_pred} ({t_conf}%) | Plant={t_plant}")
