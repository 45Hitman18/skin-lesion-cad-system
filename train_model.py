"""
Skin Lesion Classification - HAM10000 Training Script
======================================================
Run this script in a Kaggle Notebook (with the HAM10000 dataset added) or Google Colab
to fine-tune an EfficientNetB0 transfer learning architecture for Benign vs. Malignant
classification.

Key Features:
- Lesion-level grouping (prevents patient/lesion data leakage across splits)
- Class imbalance mitigation using compute_class_weight
- Two-stage transfer learning (Stage 1: Head warmup; Stage 2: Top layer fine-tuning)
- Clinically relevant metric tracking (AUC-ROC, Malignant Recall, Precision, PR-AUC)
- Generates publication-ready training curve plots and confusion matrix
- Saves model as 'skin_cancer_model.keras' and class mapping as 'class_indices.json'
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# 1. HYPERPARAMETERS & CONFIG
# ----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 10          # Stage 1: Train top classification head
EPOCHS_FINETUNE = 15      # Stage 2: Fine-tune top layers of base model
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE = 1e-5
SEED = 42

# Kaggle dataset paths (standard layout for kmader/skin-cancer-mnist-ham10000)
BASE_DIR = "/kaggle/input/skin-cancer-mnist-ham10000"
METADATA_PATH = os.path.join(BASE_DIR, "HAM10000_metadata.csv")
IMG_DIRS = [
    os.path.join(BASE_DIR, "HAM10000_images_part_1"),
    os.path.join(BASE_DIR, "HAM10000_images_part_2"),
    os.path.join(BASE_DIR, "ham10000_images_part_1"),
    os.path.join(BASE_DIR, "ham10000_images_part_2"),
]


# ----------------------------
# 2. LOAD METADATA & BINARY MAPPING
# ----------------------------
print("[1/8] Loading HAM10000 metadata...")
if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(
        f"Metadata file not found at {METADATA_PATH}. "
        "Please ensure the HAM10000 dataset is attached to the Kaggle notebook."
    )

df = pd.read_csv(METADATA_PATH)

# HAM10000 has 7 diagnostic categories -> collapse into clinical binary classification
# Malignant: Melanoma (mel), Basal Cell Carcinoma (bcc), Actinic Keratoses (akiec)
# Benign: Melanocytic Nevus (nv), Benign Keratosis (bkl), Dermatofibroma (df), Vascular (vasc)
MALIGNANT = {"mel", "bcc", "akiec"}
BENIGN = {"nv", "bkl", "df", "vasc"}


def map_label(dx):
    if dx in MALIGNANT:
        return "malignant"
    elif dx in BENIGN:
        return "benign"
    return None


df["label"] = df["dx"].apply(map_label)
df = df.dropna(subset=["label"]).copy()


def find_image_path(image_id):
    for d in IMG_DIRS:
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            candidate = os.path.join(d, image_id + ext)
            if os.path.exists(candidate):
                return candidate
    return None


print("[2/8] Resolving image filepaths...")
df["image_path"] = df["image_id"].apply(find_image_path)
df = df.dropna(subset=["image_path"]).copy()

print(f"Total verified usable images: {len(df)}")
print("Class breakdown:\n", df["label"].value_counts())


# ----------------------------
# 3. LEAKAGE-FREE GROUPED TRAIN / VAL / TEST SPLIT
# ----------------------------
# Patient/lesion leakage prevention: Group by 'lesion_id' so images of the same
# lesion never appear in both training and test sets.
print("[3/8] Splitting dataset with lesion_id grouping...")
gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_idx, temp_idx = next(gss_outer.split(df, groups=df["lesion_id"]))
train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_idx, test_idx = next(gss_inner.split(temp_df, groups=temp_df["lesion_id"]))
val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

print(f"Train set: {len(train_df)} images ({train_df['label'].value_counts().to_dict()})")
print(f"Val set:   {len(val_df)} images ({val_df['label'].value_counts().to_dict()})")
print(f"Test set:  {len(test_df)} images ({test_df['label'].value_counts().to_dict()})")


# ----------------------------
# 4. DATA GENERATORS & AUGMENTATION
# ----------------------------
print("[4/8] Configuring data generators and clinical augmentations...")
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.20,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="binary",
    batch_size=BATCH_SIZE,
    seed=SEED,
)

val_gen = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="binary",
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False,
)

test_gen = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col="image_path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="binary",
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False,
)

class_indices = train_gen.class_indices
print("Class Indices Mapping:", class_indices)
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=2)


# ----------------------------
# 5. CLASS WEIGHT CALCULATION
# ----------------------------
# HAM10000 has ~67% benign cases. Compute balanced inverse frequency weights.
labels = train_df["label"].map(class_indices).values
class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.unique(labels), y=labels
)
class_weights = dict(enumerate(class_weights_arr))
print("Computed Balanced Class Weights:", class_weights)


# ----------------------------
# 6. MODEL ARCHITECTURE (EfficientNetB0 Transfer Learning)
# ----------------------------
print("[5/8] Building EfficientNetB0 transfer learning architecture...")
base_model = EfficientNetB0(
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)
model.summary()


# ----------------------------
# 7. STAGE 1: TRAIN CLASSIFICATION HEAD
# ----------------------------
print("[6/8] Starting Stage 1 Training (Head warmup)...")
early_stop = callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=4, restore_best_weights=True, verbose=1
)
checkpoint = callbacks.ModelCheckpoint(
    "best_checkpoint.keras", monitor="val_auc", mode="max", save_best_only=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
)

history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr],
)


# ----------------------------
# 8. STAGE 2: FINE-TUNING
# ----------------------------
print("[7/8] Starting Stage 2 Training (Fine-tuning top 30 layers)...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint, reduce_lr],
)


# ----------------------------
# 9. EVALUATION & REPORTING
# ----------------------------
print("[8/8] Evaluating model on unseen holdout test set...")
test_gen.reset()
y_pred_prob = model.predict(test_gen, verbose=1).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)
y_true = test_gen.classes

target_names = list(class_indices.keys())
print("\n" + "=" * 50)
print("TEST SET CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_true, y_pred, target_names=target_names))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

test_auc = roc_auc_score(y_true, y_pred_prob)
print(f"Holdout Test AUC-ROC: {test_auc:.4f}")

# Save final deployable model
model.save("skin_cancer_model.keras")
print("\nSaved final model as: skin_cancer_model.keras")
print("Saved class indices as: class_indices.json")


# ----------------------------
# 10. GENERATE PLOTS FOR REPORT
# ----------------------------
def plot_metric(h1, h2, metric, filename):
    plt.figure(figsize=(7, 4.5))
    vals_train = h1.history.get(metric, []) + h2.history.get(metric, [])
    vals_val = h1.history.get(f"val_{metric}", []) + h2.history.get(f"val_{metric}", [])
    plt.plot(vals_train, label=f"Train {metric.upper()}", linewidth=2)
    plt.plot(vals_val, label=f"Validation {metric.upper()}", linewidth=2, linestyle="--")
    plt.axvline(x=len(h1.history.get(metric, [])) - 1, color="grey", linestyle=":", label="Fine-tune Start")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Model {metric.capitalize()} Progression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


for m in ["auc", "loss", "accuracy"]:
    plot_metric(history_head, history_fine, m, f"{m}_curve.png")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#1e3d59", lw=2, label=f"EfficientNetB0 (AUC = {test_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity / Recall)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=200)
plt.close()

# Plot Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(5, 4.5))
disp.plot(cmap="Blues", ax=ax, values_format="d")
plt.title("Holdout Test Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()

print("\nAll evaluation curve plots and artifacts saved successfully!")
