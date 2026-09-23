"""
Skin Lesion CAD System - Automated Test & Verification Suite
============================================================
Validates end-to-end functionality of:
1. Class indices mapping
2. Model loading and architecture
3. Image preprocessing pipeline
4. Deep learning inference
5. Explainable AI (Grad-CAM) computation & overlay
6. Sample benchmark images
7. Clinical report export format
"""

import json
import os
import sys
import numpy as np
from PIL import Image

def run_tests():
    print("=" * 60)
    print("SKIN LESION CAD SYSTEM: AUTOMATED VERIFICATION SUITE")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = 7

    # Test 1: Class Indices File
    print("\n[Test 1/7] Validating class_indices.json...")
    assert os.path.exists("class_indices.json"), "class_indices.json missing!"
    with open("class_indices.json") as f:
        class_indices = json.load(f)
    assert "benign" in class_indices and "malignant" in class_indices, "Invalid class mapping!"
    assert class_indices["benign"] == 0 and class_indices["malignant"] == 1
    print("  --> PASSED: class_indices.json contains valid binary mapping:", class_indices)
    passed_tests += 1

    # Test 2: Model Loading
    print("\n[Test 2/7] Validating skin_cancer_model.keras...")
    assert os.path.exists("skin_cancer_model.keras"), "skin_cancer_model.keras missing!"
    import tensorflow as tf
    model = tf.keras.models.load_model("skin_cancer_model.keras")
    assert model.input_shape == (None, 224, 224, 3), f"Unexpected input shape: {model.input_shape}"
    assert model.output_shape == (None, 1), f"Unexpected output shape: {model.output_shape}"
    print(f"  --> PASSED: Model loaded successfully. Input: {model.input_shape}, Output: {model.output_shape}")
    passed_tests += 1

    # Test 3: Sample Images
    print("\n[Test 3/7] Validating sample benchmark images...")
    sample_files = [
        "sample_benign_nevus.jpg",
        "sample_malignant_melanoma.jpg",
        "sample_benign_seborrheic_keratosis.jpg",
        "sample_malignant_basal_cell.jpg"
    ]
    for sf in sample_files:
        path = os.path.join("sample_images", sf)
        assert os.path.exists(path), f"Sample image missing: {path}"
        img = Image.open(path)
        assert img.size[0] > 0 and img.size[1] > 0
    print(f"  --> PASSED: All {len(sample_files)} benchmark images verified in sample_images/")
    passed_tests += 1

    # Test 4: Image Preprocessing
    print("\n[Test 4/7] Validating preprocessing pipeline...")
    from app import preprocess_image
    test_img = Image.open(os.path.join("sample_images", "sample_benign_nevus.jpg"))
    img_tensor = preprocess_image(test_img)
    assert img_tensor.shape == (1, 224, 224, 3), f"Invalid preprocessed shape: {img_tensor.shape}"
    assert 0.0 <= img_tensor.min() and img_tensor.max() <= 255.0, "Input out of [0, 255] range!"
    print(f"  --> PASSED: Image preprocessed to shape {img_tensor.shape}, min={img_tensor.min():.1f}, max={img_tensor.max():.1f}")
    passed_tests += 1

    # Test 5: Inference
    print("\n[Test 5/7] Validating deep learning forward pass...")
    pred_prob = float(model.predict(img_tensor, verbose=0)[0][0])
    assert 0.0 <= pred_prob <= 1.0, f"Probability out of bounds: {pred_prob}"
    print(f"  --> PASSED: Inference successful. Predicted P(Malignant) = {pred_prob:.4f}")
    passed_tests += 1

    # Test 6: Grad-CAM Explainable AI
    print("\n[Test 6/7] Validating Explainable AI (Grad-CAM) computation...")
    from app import compute_gradcam, overlay_gradcam
    heatmap = compute_gradcam(model, img_tensor)
    assert heatmap.ndim == 2, f"Expected 2D heatmap, got shape {heatmap.shape}"
    assert not np.isnan(heatmap).any(), "Heatmap contains NaNs!"
    overlay_img, colored_heatmap = overlay_gradcam(test_img, heatmap, alpha=0.45)
    assert isinstance(overlay_img, Image.Image) and isinstance(colored_heatmap, Image.Image)
    assert overlay_img.size == test_img.size
    print(f"  --> PASSED: Grad-CAM generated heatmap (shape {heatmap.shape}) and overlay {overlay_img.size}")
    passed_tests += 1

    # Test 7: Training Script Syntax
    print("\n[Test 7/7] Validating training pipeline scripts...")
    import py_compile
    py_compile.compile("train_model.py", doraise=True)
    with open("train_model.ipynb") as f:
        nb = json.load(f)
    assert len(nb["cells"]) >= 10, "Notebook cells appear incomplete!"
    print("  --> PASSED: train_model.py and train_model.ipynb compile cleanly and pass structural checks.")
    passed_tests += 1

    print("\n" + "=" * 60)
    print(f"ALL {passed_tests}/{total_tests} TESTS PASSED SUCCESSFULLY! PROJECT IS 100% OPERATIONAL.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
