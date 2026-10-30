# Leaf-disease-prediction

## FloraScan AI

**FloraScan AI** is an intelligent, end-to-end computer vision and deep learning platform engineered to identify plant species and diagnose foliar diseases from user-uploaded leaf photographs. The system focuses on three high-impact, economically vital crops:

- **Bell Pepper**
- **Potato**
- **Tomato**

## Architecture and Diagnostic Capabilities

FloraScan AI uses transfer learning with an optimized **MobileNetV2** backbone, pre-trained on ImageNet and fine-tuned on standardized plant disease datasets such as PlantVillage. The classifier operates across nine classes:

- **Bell Pepper**
  - Healthy
  - Bacterial Spot (*Xanthomonas campestris*)
- **Potato**
  - Healthy
  - Early Blight (*Alternaria solani*)
  - Late Blight (*Phytophthora infestans*)
- **Tomato**
  - Healthy
  - Bacterial Spot
  - Early Blight
  - Late Blight

A single-stage model predicts both crop identity and disease condition in one pass. Input images are resized to **224 × 224** and normalized with ImageNet statistics before inference.

## Real-Time Inference and Agronomic Guidance

To enable fast, low-cost deployment on commodity CPUs, trained weights are exported to **ONNX** and served through an **ONNX Runtime** inference engine integrated with a lightweight **FastAPI** backend.

The platform is designed to deliver:

1. **Confidence metrics** with class probabilities and top-3 predictions.
2. **Pathogen breakdown** that distinguishes fungal, bacterial, and oomycete causes.
3. **Actionable treatments** including options such as copper bactericides, chlorothalonil, or systemic fungicides.
4. **Preventive strategies** such as crop rotation, drip irrigation, and canopy aeration to reduce recurrence.
