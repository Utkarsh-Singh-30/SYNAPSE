# SYNAPSE: Advanced Deep Learning Ensemble for Thoracic Pathology Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging%20(Radiology)-10B981)
![Status](https://img.shields.io/badge/Status-v1.0%20Baseline-success)

## 📌 Overview
**SYNAPSE** is a robust, clinical-grade deep learning framework designed for the automated multi-label classification of chest radiographs (X-rays). Medical imaging analysis requires systems that are not only accurate but also inherently safe. SYNAPSE is built on a **"Safety-First / High-Recall"** philosophy, ensuring that potential thoracic pathologies are aggressively flagged during initial diagnostic screenings, thereby significantly minimizing the risk of False Negatives.

The system evaluates chest X-rays against **14 distinct thoracic diseases** simultaneously, acting as an automated second opinion to assist radiologists and reduce diagnostic bottlenecks in high-volume clinical settings.

---

## 🔬 Dataset: The Foundation
SYNAPSE is foundationaly trained and validated on the highly regarded **NIH Chest X-ray Dataset**. 
* **Scale:** Comprises over 100,000 anonymized frontal-view X-ray images.
* **Multi-Label Complexity:** Patients in the real world often suffer from multiple conditions simultaneously (e.g., Pneumonia *and* Effusion). The dataset and our loss functions (`BCEWithLogitsLoss`) are structured to handle this multi-label reality.
* **Target Pathologies (14 Classes):**
  Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, and Hernia.

---

## 🧠 Core Architecture: The Ensemble approach
No single neural network architecture is perfect. To mitigate individual model biases and improve diagnostic stability, SYNAPSE utilizes a **Soft-Voting Ensemble** of three diverse Convolutional Neural Networks (CNNs). 



1. **DenseNet121 (The Feature Extractor):**
   * *Why:* Dense connections ensure maximum information flow between layers.
   * *Role:* Highly efficient at capturing fine-grained, subtle pathological textures (like micro-nodules or light infiltrations) that are easily missed by standard networks.
2. **ResNet101 (The Structural Analyst):**
   * *Why:* Deep residual learning prevents the vanishing gradient problem.
   * *Role:* Excels at understanding global context and large structural abnormalities, such as an enlarged heart (Cardiomegaly) or major lung consolidations.
3. **SE-ResNeXt50 (The Attention Mechanism):**
   * *Why:* Integrates Squeeze-and-Excitation (SE) blocks to adaptively recalibrate channel-wise feature responses.
   * *Role:* Acts as a visual attention mechanism, explicitly learning *which* features are important for a specific disease and suppressing irrelevant background noise (like bones or medical equipment).

---

## 🛡️ Key Innovation: The Gatekeeper Module
In a real-world deployment (e.g., a web portal or hospital app), users might accidentally upload invalid images (selfies, documents, or blurry scans). Running heavy deep learning models on garbage data wastes server resources and produces dangerous hallucinated predictions.

To solve this, SYNAPSE implements a lightweight, heuristic **Gatekeeper Pre-processing Module** that acts as an input firewall:
* **Color Check (HSV Saturation):** Instantly rejects RGB/Colorful images, as authentic medical X-rays are strictly grayscale.
* **Blur & Focus Check (Laplacian Variance):** Measures image sharpness. Scans falling below a critical variance threshold are rejected to prevent misdiagnosis from out-of-focus inputs.
* **Texture & Edge Density:** Analyzes the edge map of the image. Highly textured images (like outdoor scenery) are flagged and blocked, ensuring only smooth, bone/tissue-like structures pass through.

---

## ⚙️ Inference Pipeline & Clinical Thresholding
When a valid X-ray passes the Gatekeeper, the inference pipeline follows strict clinical rules:
1. **Pre-processing:** Image is resized to standard dimensions, center-cropped, and normalized using ImageNet statistics.
2. **Parallel Inference:** The image is fed into all three ensemble models simultaneously.
3. **Soft Voting Aggregation:** The Sigmoid probability outputs (ranging from 0 to 1 for all 14 classes) from each model are mathematically averaged.
4. **Safety-First Thresholding (0.4):** While standard machine learning defaults to a 0.5 threshold, SYNAPSE utilizes a **0.4 sensitivity threshold**. In the medical domain, over-flagging a healthy patient (False Positive) requires a simple re-check, but missing a diseased patient (False Negative) can be fatal. This threshold guarantees exceptionally high Recall.

---

## 🛠️ Technology Stack
* **Deep Learning Framework:** PyTorch, Torchvision
* **Advanced Architectures:** TIMM (PyTorch Image Models)
* **Data Processing Pipeline:** Pandas, NumPy, Pillow (PIL)
* **Evaluation Metrics:** Scikit-Learn (Macro-Average ROC-AUC, Recall/Sensitivity, Accuracy)
* **Hardware Optimization:** Mixed Precision (AMP) for VRAM-efficient training.

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed along with a CUDA-enabled GPU for optimal inference speed.
```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install timm pandas numpy pillow scikit-learn tqdm
