# 🩺 Skin Lesion CAD Diagnostic System (Benign vs. Malignant)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end clinical **Computer-Aided Diagnosis (CAD)** web application for dermatological lesion assessment. Built with **EfficientNetB0 Transfer Learning**, **Explainable AI (Grad-CAM)** saliency overlays, **ABCDE clinical rule** scoring, adjustable diagnostic sensitivity thresholds, and automated clinical CAD diagnostic report generation.

---

## 🌟 Key Features

1. **Deep Learning Classifier**:
   - **Backbone**: EfficientNetB0 pre-trained on ImageNet (5.3M parameters).
   - **Custom Head**: Global Average Pooling, Dropout (0.3 / 0.2), Dense (128, ReLU), and Sigmoid output calibrated for malignancy probability $P(\text{Malignant})$.
   - **Trained on HAM10000**: Human Against Machine with 10,000 dermatoscopic images.

2. **Explainable AI (Grad-CAM Visualizations)**:
   - Real-time gradient-weighted class activation mapping directly computed on the top convolutional feature maps (`top_conv`).
   - Superimposed JET colormap heatmaps highlighting pigmented networks, border irregularities, and vascular telangiectasias.
   - Interactive alpha opacity slider.

3. **Clinical Decision Support & Risk Tiers**:
   - **Adjustable Sensitivity Threshold ($\tau$)**: Shift between High-Sensitivity Screening ($\tau = 0.35$, minimizes false negatives) and High-Specificity ($\tau = 0.65$).
   - **Tri-level Risk Classification**:
     - 🟢 **Low Risk** ($P < 0.30$) — Routine periodic observation.
     - 🟡 **Intermediate / Indeterminate** ($0.30 \le P \le 0.60$) — Close 3-month dermoscopic follow-up.
     - 🔴 **High Suspicion** ($P > 0.60$) — Urgent dermato-oncology consultation and biopsy alert.

4. **Dermatological ABCDE Criteria Checklist**:
   - Interactive evaluation of Asymmetry, Border irregularity, Color variegation, Diameter (>6mm), and Evolution.
   - Generates composite clinical risk scores correlating physical signs with deep learning predictions.

5. **Pre-Loaded Benchmark Lesion Gallery**:
   - Built-in 1-click test dermoscopic images:
     - 🟤 Benign Melanocytic Nevus (Common Mole)
     - 🟣 Malignant Melanoma
     - 🟡 Benign Seborrheic Keratosis
     - 🔴 Malignant Basal Cell Carcinoma (BCC)

6. **Instant Downloadable Patient/CAD Report**:
   - Generates a clinical summary text report containing timestamp, model parameters, probabilities, risk tier, and recommendations.

---

## 📂 Project Structure

```
├── app.py                      # Main Streamlit web application & Grad-CAM pipeline
├── class_indices.json          # Binary class mapping: {"benign": 0, "malignant": 1}
├── skin_cancer_model.keras     # Ready-to-deploy EfficientNetB0 Keras model weights (~17.7 MB)
├── requirements.txt            # Application dependencies for local and cloud deployment
├── sample_images/              # Benchmark dermoscopic test images for 1-click evaluation
│   ├── sample_benign_nevus.jpg
│   ├── sample_malignant_melanoma.jpg
│   ├── sample_benign_seborrheic_keratosis.jpg
│   └── sample_malignant_basal_cell.jpg
├── test_cad_system.py          # Automated 7-point verification test suite
├── train_model.py              # Full training script with grouped patient splitting
├── train_model.ipynb           # Kaggle / Google Colab interactive training notebook
└── README.md                   # Comprehensive documentation and deployment guide
```

---

## 🚀 Quick Start (Run Locally)

### 1. Clone or Open Workspace
```bash
git clone <your-repo-url>
cd Assignment
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the CAD Web Application
```bash
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 🧪 Run Automated Verification Tests

Run the test suite to verify model integrity, tensor shapes, preprocessing, Grad-CAM generation, and sample images:

```bash
python test_cad_system.py
```
Expected output:
```
============================================================
SKIN LESION CAD SYSTEM: AUTOMATED VERIFICATION SUITE
============================================================
[Test 1/7] Validating class_indices.json... --> PASSED
[Test 2/7] Validating skin_cancer_model.keras... --> PASSED
[Test 3/7] Validating sample benchmark images... --> PASSED
[Test 4/7] Validating preprocessing pipeline... --> PASSED
[Test 5/7] Validating deep learning forward pass... --> PASSED
[Test 6/7] Validating Explainable AI (Grad-CAM)... --> PASSED
[Test 7/7] Validating training pipeline scripts... --> PASSED
============================================================
ALL 7/7 TESTS PASSED SUCCESSFULLY! PROJECT IS 100% OPERATIONAL.
============================================================
```

---

## ☁️ Free Public Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "Deploy Skin Lesion CAD System"
   git push origin main
   ```
   *(Note: The bundled `skin_cancer_model.keras` is ~17.7 MB, well within GitHub's 100MB file limit.)*

2. Visit **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **New app**, select your repository, branch (`main`), and set Main file path to `app.py`.
4. Click **Deploy**.
5. You will receive a public URL (e.g. `https://<your-username>-skin-cad.streamlit.app`) to share or submit.

---

## 🔬 Model Training & Methodology (Kaggle / Colab)

To reproduce or re-train the model on the full HAM10000 dataset:

### Option A: Kaggle Notebook (Recommended for Free GPUs)
1. Go to [kaggle.com](https://www.kaggle.com/) and create a **New Notebook**.
2. Click **Add Input** → search for `Skin Cancer MNIST: HAM10000` (by kmader) → add it.
3. Turn on GPU: **Settings → Accelerator → GPU T4 x2**.
4. Upload or paste `train_model.ipynb` (or copy `train_model.py`) and run all cells.
5. Download `skin_cancer_model.keras`, `class_indices.json`, and the generated metric curves (`roc_curve.png`, `confusion_matrix.png`, `auc_curve.png`).

### Critical Coursework & Clinical Rationale:
- **Binary Label Aggregation**:
  - **Malignant**: Melanoma (`mel`), Basal Cell Carcinoma (`bcc`), Actinic Keratoses (`akiec`).
  - **Benign**: Melanocytic Nevi (`nv`), Benign Keratoses (`bkl`), Dermatofibroma (`df`), Vascular Lesions (`vasc`).
- **Eliminating Patient Data Leakage (`lesion_id` Grouping)**:
  - In HAM10000, multiple images frequently originate from the same lesion. Splitting purely randomly across images leaks features into the test set. We use `GroupShuffleSplit` on `lesion_id` to ensure complete independence between splits.
- **Handling Class Imbalance**:
  - Benign nevi (`nv`) constitute ~67% of the dataset. We compute inverse frequency class weights (`balanced`) to prevent the model from defaulting to the majority class.
- **Prioritizing AUC-ROC and Malignant Recall**:
  - In oncological triage, a false negative (failing to detect melanoma) is critical. Accuracy is misleading on imbalanced medical datasets; hence, AUC-ROC and Malignant Recall are the primary optimization metrics.

---

## 📄 License & Academic Disclaimer

This project is released under the MIT License. It was developed for coursework, academic demonstration, and clinical decision support research. It is **not** an FDA-approved medical device and is not a substitute for professional dermatologist consultation or histopathological biopsy.
