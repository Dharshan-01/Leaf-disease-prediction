"""
Export PyTorch EfficientNet-B0 checkpoint to optimized ONNX format.
Exports both logits (output) and 1280-dim feature embedding (embedding)
for real-time disease diagnosis and out-of-distribution leaf morphology verification.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from model import create_model


class EfficientNetDualOutputWrapper(nn.Module):
    """Wraps model to output both classification logits and 1280-dim feature embeddings."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        feats = self.model.extract_features(x)
        logits = self.model.backbone.classifier(feats)
        return logits, feats


def export_to_onnx(
    checkpoint_path: str = "best_model.pth",
    output_onnx_path: str = "model.onnx",
    num_classes: int = 9,
    img_size: int = 224
):
    print("=" * 60)
    print("EXPORTING EFFICIENTNET-B0 MODEL TO DUAL-OUTPUT ONNX")
    print("=" * 60)
    
    device = torch.device("cpu")
    model = create_model(num_classes=num_classes, pretrained=False)
    
    ckpt_file = Path(checkpoint_path)
    if ckpt_file.exists():
        print(f"Loading trained weights from: {ckpt_file.resolve()}")
        checkpoint = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"[!] Warning: '{checkpoint_path}' not found.")

    model.eval()
    wrapper = EfficientNetDualOutputWrapper(model)

    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output", "embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
            "embedding": {0: "batch_size"}
        },
        dynamo=False
    )
    print(f"Model exported successfully to: {Path(output_onnx_path).resolve()}")

    # 1. Validate ONNX model graph structure
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure verified: OK")

    # 2. Test inference with ONNX Runtime
    ort_session = ort.InferenceSession(output_onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # 3. Compare with PyTorch output
    with torch.no_grad():
        torch_logits, torch_feats = wrapper(dummy_input)

    diff_logits = np.max(np.abs(torch_logits.numpy() - ort_outputs[0]))
    diff_feats = np.max(np.abs(torch_feats.numpy() - ort_outputs[1]))
    print(f"Max absolute difference (Logits): {diff_logits:.6e}")
    print(f"Max absolute difference (Embedding): {diff_feats:.6e}")
    if diff_logits < 1e-4 and diff_feats < 1e-4:
        print("Model numerical validation PASSED!")


if __name__ == "__main__":
    export_to_onnx()
