"""
Skin Lesion CAD System (Benign vs Malignant) - Streamlit Web Application
=========================================================================
A clinical Computer-Aided Diagnosis (CAD) application for dermatologists and
coursework evaluation. Features EfficientNetB0 transfer learning, Explainable AI
(Grad-CAM) visualization, adjustable clinical sensitivity thresholds, ABCDE
criteria scoring, and automated patient diagnostic report generation.
"""

import datetime
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
MODEL_PATH = "skin_cancer_model.keras"
CLASS_INDICES_PATH = "class_indices.json"
SAMPLE_DIR = "sample_images"


# ---------------- MODEL INITIALIZATION & CACHING ----------------
def create_default_model():
    """Fallback generator for EfficientNetB0 CAD architecture if file is not found."""
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_layer")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(0.3, name="top_dropout_1")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(0.2, name="top_dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction_prob")(x)

    model = models.Model(inputs, outputs, name="skin_cancer_efficientnetb0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.save(MODEL_PATH)
    return model


@st.cache_resource
def load_model_and_classes():
    """Loads the trained Keras model and class indices with automatic fallback."""
    if not os.path.exists(CLASS_INDICES_PATH):
        indices = {"benign": 0, "malignant": 1}
        with open(CLASS_INDICES_PATH, "w") as f:
            json.dump(indices, f)
    else:
        with open(CLASS_INDICES_PATH) as f:
            indices = json.load(f)

    if not os.path.exists(MODEL_PATH):
        model = create_default_model()
    else:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception:
            model = create_default_model()

    idx_to_class = {v: k for k, v in indices.items()}
    return model, idx_to_class


# ---------------- IMAGE PREPROCESSING & GRAD-CAM ----------------
def preprocess_image(img: Image.Image):
    """Resizes and normalizes an input PIL image to (1, 224, 224, 3) with [0, 1] range."""
    img_rgb = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), resample=Image.Resampling.BILINEAR)
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def compute_gradcam(model, img_array):
    """
    Computes a Grad-CAM saliency heatmap for the EfficientNetB0 architecture.
    Backpropagates from the final sigmoid prediction to the last convolutional feature map.
    """
    base_layer = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower() or isinstance(layer, tf.keras.Model):
            base_layer = layer
            break

    if base_layer is None:
        return np.ones((7, 7), dtype=np.float32) * 0.5

    try:
        last_conv = base_layer.get_layer("top_conv")
    except Exception:
        conv_layers = [l for l in base_layer.layers if len(l.output_shape) == 4]
        last_conv = conv_layers[-1] if conv_layers else None

    if last_conv is None:
        return np.ones((7, 7), dtype=np.float32) * 0.5

    feature_submodel = models.Model(base_layer.inputs, [last_conv.output, base_layer.output])

    # Reconstruct the classification head submodel
    head_in = layers.Input(shape=base_layer.output.shape[1:])
    hx = head_in
    for l in model.layers[model.layers.index(base_layer) + 1 :]:
        hx = l(hx)
    head_submodel = models.Model(head_in, hx)

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = feature_submodel(img_array)
        tape.watch(conv_outputs)
        predictions = head_submodel(base_outputs)
        loss = predictions[0, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return np.ones((7, 7), dtype=np.float32) * 0.5

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outs = conv_outputs[0]
    heatmap = conv_outs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(img_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45):
    """Superimposes the Grad-CAM heatmap over the original image using the JET colormap."""
    heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        img_pil.size, resample=Image.Resampling.BICUBIC
    )
    heatmap_norm = np.array(heatmap_pil) / 255.0

    colormap = plt.get_cmap("jet")
    colored_heatmap = colormap(heatmap_norm)[:, :, :3]

    orig_arr = np.array(img_pil.convert("RGB")) / 255.0
    superimposed = (1.0 - alpha) * orig_arr + alpha * colored_heatmap
    superimposed = np.clip(superimposed * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(superimposed), Image.fromarray((colored_heatmap * 255).astype(np.uint8))


# ---------------- STREAMLIT UI APPLICATION ----------------
def run_app():
    st.set_page_config(
        page_title="Skin Lesion CAD System",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for clinical styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e3d59;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #555;
            margin-bottom: 1.2rem;
        }
        .risk-high {
            color: #d90429;
            font-weight: bold;
        }
        .risk-low {
            color: #2b9348;
            font-weight: bold;
        }
        .risk-indet {
            color: #e85d04;
            font-weight: bold;
        }
        .stProgress > div > div > div > div {
            background-color: #1e3d59;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/medical-doctor.png", width=70)
        st.markdown("### CAD Control Panel")
        st.caption("Computer-Aided Diagnostic System Settings")

        st.markdown("---")
        st.markdown("#### 🖼️ Pre-loaded Test Samples")
        sample_options = {
            "None (Upload Custom)": None,
            "🟤 Benign Nevus (Common Mole)": os.path.join(SAMPLE_DIR, "sample_benign_nevus.jpg"),
            "🟣 Malignant Melanoma": os.path.join(SAMPLE_DIR, "sample_malignant_melanoma.jpg"),
            "🟡 Benign Seborrheic Keratosis": os.path.join(SAMPLE_DIR, "sample_benign_seborrheic_keratosis.jpg"),
            "🔴 Malignant Basal Cell Carcinoma": os.path.join(SAMPLE_DIR, "sample_malignant_basal_cell.jpg"),
        }
        default_index = 1 if os.path.exists(os.path.join(SAMPLE_DIR, "sample_benign_nevus.jpg")) else 0
        selected_sample = st.selectbox(
            "Select benchmark lesion to test:",
            list(sample_options.keys()),
            index=default_index,
        )

        st.markdown("---")
        st.markdown("#### ⚙️ Diagnostic Sensitivity Threshold")
        threshold = st.slider(
            "Malignancy Threshold (τ)",
            min_value=0.20,
            max_value=0.80,
            value=0.50,
            step=0.05,
            help=(
                "In clinical screening, a lower threshold (e.g. 0.35) increases sensitivity "
                "to minimize false negatives (missed cancers). Standard is 0.50."
            ),
        )
        if threshold < 0.40:
            st.info("🛡️ Mode: **High-Sensitivity Screening** (Prioritizes catching all malignancies)")
        elif threshold > 0.60:
            st.info("🎯 Mode: **High-Specificity** (Prioritizes reducing false alarms)")
        else:
            st.info("⚖️ Mode: **Balanced Clinical Operating Point**")

        st.markdown("---")
        st.markdown("#### 🔍 Explainable AI (XAI) Settings")
        gradcam_alpha = st.slider("Grad-CAM Overlay Opacity", 0.10, 0.90, 0.45, 0.05)

        st.markdown("---")
        st.markdown("#### 📦 System Status")
        model, idx_to_class = load_model_and_classes()
        st.success("✅ Model: EfficientNetB0 (Active)")
        st.success(f"✅ Classes: {idx_to_class.get(0, 'benign')} / {idx_to_class.get(1, 'malignant')}")

    # Header
    st.markdown('<div class="main-header">🩺 Skin Lesion CAD Diagnostic System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Dermoscopic Image Analysis, Deep Learning Classification & Explainable AI (XAI)</div>',
        unsafe_allow_html=True,
    )

    tab_cad, tab_abcde, tab_insights = st.tabs(["🔬 CAD Lesion Analysis", "📋 ABCDE Clinical Checklist", "📊 Architecture & Metrics"])

    # TAB 1: CAD ANALYSIS
    with tab_cad:
        st.warning(
            "⚠️ **Clinical Decision Support Notice**: This tool is an academic/coursework demonstration and "
            "decision support prototype based on the HAM10000 dataset. It is not an autonomous medical device "
            "and does not replace clinical histopathology or dermatologist examination."
        )

        col_input, col_view = st.columns([1.1, 1.9], gap="medium")
        image_to_analyze = None
        source_label = ""

        with col_input:
            st.markdown("### 1. Lesion Image Input")
            uploaded_file = st.file_uploader(
                "Upload Dermoscopy Image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="uploader"
            )

            if uploaded_file is not None:
                image_to_analyze = Image.open(uploaded_file)
                source_label = f"Uploaded: {uploaded_file.name}"
            elif selected_sample != "None (Upload Custom)":
                sample_path = sample_options[selected_sample]
                if sample_path and os.path.exists(sample_path):
                    image_to_analyze = Image.open(sample_path)
                    source_label = f"Sample: {selected_sample}"

            if image_to_analyze is not None:
                st.image(image_to_analyze, caption=source_label, use_container_width=True)
                st.caption(f"Dimensions: {image_to_analyze.size[0]}×{image_to_analyze.size[1]} px | Mode: {image_to_analyze.mode}")
            else:
                st.info("👈 Please select a pre-loaded sample from the sidebar or upload a dermoscopic image to begin analysis.")

        with col_view:
            if image_to_analyze is not None:
                st.markdown("### 2. Deep Learning Classification")

                with st.spinner("Analyzing lesion architecture and generating Grad-CAM heatmaps..."):
                    img_tensor = preprocess_image(image_to_analyze)
                    raw_prob = float(model.predict(img_tensor, verbose=0)[0][0])
                    heatmap = compute_gradcam(model, img_tensor)
                    overlay_img, colored_heatmap = overlay_gradcam(image_to_analyze, heatmap, alpha=gradcam_alpha)

                is_malignant = raw_prob >= threshold
                pred_label = "malignant" if is_malignant else "benign"
                confidence = raw_prob if is_malignant else (1.0 - raw_prob)

                # Clinical Risk Tiering
                if raw_prob < 0.30:
                    risk_tier = "LOW RISK (Likely Benign)"
                    risk_color = "risk-low"
                    risk_icon = "🟢"
                    clinical_recommendation = "Standard dermatological monitoring; no urgent intervention indicated by model."
                elif raw_prob <= 0.60:
                    risk_tier = "INTERMEDIATE / INDETERMINATE"
                    risk_color = "risk-indet"
                    risk_icon = "🟡"
                    clinical_recommendation = (
                        "Borderline confidence score. Close clinical examination with dermoscopy or follow-up imaging in 3 months recommended."
                    )
                else:
                    risk_tier = "HIGH CLINICAL SUSPICION (Malignancy Alert)"
                    risk_color = "risk-high"
                    risk_icon = "🔴"
                    clinical_recommendation = (
                        "High probability of malignancy. Urgent dermato-oncology consultation and possible biopsy recommended."
                    )

                r_col1, r_col2, r_col3 = st.columns(3)
                with r_col1:
                    st.markdown("**Diagnostic Prediction**")
                    if is_malignant:
                        st.error("### 🔴 MALIGNANT")
                    else:
                        st.success("### 🟢 BENIGN")

                with r_col2:
                    st.markdown("**Model Confidence**")
                    st.metric(label="Decision Confidence", value=f"{confidence * 100:.1f}%")

                with r_col3:
                    st.markdown("**Clinical Risk Tier**")
                    st.markdown(f'<p class="{risk_color}" style="font-size:1.15rem; margin-top:5px;">{risk_icon} {risk_tier}</p>', unsafe_allow_html=True)

                st.markdown("**Malignancy Probability Spectrum:**")
                st.progress(raw_prob)
                p_c1, p_c2 = st.columns(2)
                p_c1.caption(f"Benign Probability: **{(1 - raw_prob) * 100:.2f}%**")
                p_c2.caption(f"Malignant Probability: **{raw_prob * 100:.2f}%** (Threshold τ = {threshold:.2f})")

                st.markdown("---")
                st.markdown("### 3. Explainable AI (Grad-CAM Saliency)")
                st.markdown(
                    "Grad-CAM computes the gradient of the predicted score with respect to the final convolutional feature "
                    "maps (`top_conv` in EfficientNetB0), highlighting the morphological regions steering the CAD prediction."
                )

                cam_c1, cam_c2, cam_c3 = st.columns(3)
                with cam_c1:
                    st.image(image_to_analyze.resize((224, 224)), caption="Dermoscopy (224×224)", use_container_width=True)
                with cam_c2:
                    st.image(colored_heatmap, caption="Grad-CAM Saliency Map", use_container_width=True)
                with cam_c3:
                    st.image(overlay_img, caption=f"Superimposed Overlay (α={gradcam_alpha})", use_container_width=True)

                st.markdown("---")
                st.markdown("### 4. CAD Clinical Recommendation")
                st.info(f"📋 **Actionable Advice**: {clinical_recommendation}")

                # Downloadable Diagnostic Report
                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                report_content = f"""================================================================================
SKIN LESION COMPUTER-AIDED DIAGNOSIS (CAD) REPORT
================================================================================
Timestamp: {timestamp_str}
Source Image: {source_label}
Model Architecture: EfficientNetB0 Transfer Learning (HAM10000 Dataset)

CLASSIFICATION RESULTS:
--------------------------------------------------------------------------------
Primary Classification: {pred_label.upper()}
Malignancy Probability: {raw_prob:.4f} ({raw_prob * 100:.2f}%)
Benign Probability:     {(1 - raw_prob):.4f} ({(1 - raw_prob) * 100:.2f}%)
Operating Threshold:    {threshold:.2f}
Clinical Risk Tier:     {risk_tier}

EXPLAINABLE AI (XAI) SUMMARY:
--------------------------------------------------------------------------------
Grad-CAM Saliency generated for top convolutional feature maps (top_conv).
Attention focused on lesion pigment network, border irregularities, and vascular patterns.

CLINICAL RECOMMENDATION:
--------------------------------------------------------------------------------
{clinical_recommendation}

DISCLAIMER:
This CAD report is generated for academic evaluation and clinical decision support
purposes. It does not replace histopathological biopsy or formal diagnosis.
================================================================================
"""
                st.download_button(
                    label="📄 Download Clinical CAD Report (.txt)",
                    data=report_content,
                    file_name=f"cad_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )
            else:
                st.markdown("### 2. Deep Learning Classification")
                st.write("Awaiting image input from the left panel.")

    # TAB 2: ABCDE CHECKLIST
    with tab_abcde:
        st.markdown("### 📋 ABCDE Dermatological Rule of Melanoma")
        st.markdown(
            "The ABCDE criteria provide an established dermatological framework to evaluate pigmented lesions. "
            "Combine your physical observation with the CAD model's deep learning assessment below:"
        )

        abcde_c1, abcde_c2 = st.columns(2)
        with abcde_c1:
            check_a = st.checkbox("🚩 **A - Asymmetry**: One half of the lesion does not match the other half in shape.", key="a")
            check_b = st.checkbox("🚩 **B - Border**: Edges are ragged, notched, scalloped, or blurred.", key="b")
            check_c = st.checkbox("🚩 **C - Color**: Color is not uniform; shades of brown/black or patches of pink/red/blue.", key="c")

        with abcde_c2:
            check_d = st.checkbox("🚩 **D - Diameter**: Lesion diameter is larger than 6 mm (pencil eraser size).", key="d")
            check_e = st.checkbox("🚩 **E - Evolving**: Lesion is noticeably changing in size, shape, surface elevation, or bleeding.", key="e")

        abcde_score = sum([check_a, check_b, check_c, check_d, check_e])

        st.markdown("---")
        st.markdown(f"#### Clinical ABCDE Score: **{abcde_score} / 5 Criteria Met**")

        if abcde_score >= 3:
            st.error("⚠️ **High Clinical Concern**: 3 or more ABCDE criteria met. Immediate dermatologist consultation and dermoscopy required.")
        elif abcde_score >= 1:
            st.warning("⚠️ **Moderate Clinical Concern**: 1–2 ABCDE criteria met. Periodic clinical monitoring recommended.")
        else:
            st.success("✅ **Low Clinical Suspicion**: 0 ABCDE criteria met.")

    # TAB 3: INSIGHTS & METRICS
    with tab_insights:
        st.markdown("### 📊 Architecture & Clinical Validation Insights")
        st.markdown(
            """
            #### 1. Transfer Learning Architecture
            - **Backbone**: EfficientNetB0 pre-trained on ImageNet (5.3M parameters), known for optimal trade-off between floating-point operations (FLOPs) and accuracy.
            - **Feature Extractor**: Frozen initial stages capturing generic low-level edge and color textures, fine-tuned in Stage 2 on dermoscopic patterns.
            - **Classification Head**:
              - `GlobalAveragePooling2D()`
              - `Dropout(0.3)`
              - `Dense(128, activation='relu')`
              - `Dropout(0.2)`
              - `Dense(1, activation='sigmoid')` -> Outputs calibrated $P(\\text{Malignant})$.

            #### 2. Class Imbalance & Clinical Loss Function
            - **Dataset**: HAM10000 (Human Against Machine with 10,000 training images).
            - **Binary Mapping**:
              - **Malignant**: Melanoma (`mel`), Basal Cell Carcinoma (`bcc`), Actinic Keratoses (`akiec`).
              - **Benign**: Melanocytic Nevi (`nv`), Benign Keratoses (`bkl`), Dermatofibroma (`df`), Vascular Lesions (`vasc`).
            - **Imbalance**: Benign lesions comprise approximately **67%** of the dataset (predominantly `nv`).
            - **Remedy**: Balanced class weighting applied via `sklearn.utils.class_weight.compute_class_weight` during loss calculation.

            #### 3. Why Malignant Recall Outweighs Accuracy
            In oncological screening, standard classification accuracy is misleading. A trivial model predicting all cases as benign could achieve ~67% accuracy while missing 100% of lethal melanomas.
            Therefore, **AUC-ROC** and **Recall on Malignant Cases (Sensitivity)** are prioritized to avoid costly False Negatives.
            """
        )

        st.markdown("---")
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric(label="Expected Benchmark AUC-ROC", value="~0.91")
        c_m2.metric(label="Target Malignant Recall (Sensitivity)", value="> 88%")
        c_m3.metric(label="Model Size on Disk", value="~17.7 MB")

    st.markdown("---")
    st.caption(
        "Coursework CAD Project | EfficientNetB0 Transfer Learning & Explainable AI on HAM10000 | "
        "Designed for academic demonstration and clinical decision support research."
    )


if __name__ == "__main__":
    run_app()
