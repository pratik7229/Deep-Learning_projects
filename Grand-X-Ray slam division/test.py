import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = Path("/Users/pratik/Documents/Projects/kaggleCompetation data new/grand-xray-slam-division-b")
MODEL_PATH = BASE_DIR / "best_xray_model.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# ==========================================
# Auto-detect test image folder
# ==========================================
possible_test_dirs = ["test2", "test", "images_test", "test_images"]
TEST_IMAGE_DIR = None
for d in possible_test_dirs:
    p = BASE_DIR / d
    if p.exists() and p.is_dir():
        TEST_IMAGE_DIR = p
        break

if TEST_IMAGE_DIR is None:
    raise FileNotFoundError(f"No test image folder found in {BASE_DIR}. Checked: {possible_test_dirs}")

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
print("GPU Devices:", tf.config.list_physical_devices("GPU"))
print("Using test image folder:", TEST_IMAGE_DIR)

# ==========================================
# Load model
# ==========================================
print("Loading model...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded from:", MODEL_PATH)

# ==========================================
# Build test file list directly from folder
# (No test CSV needed)
# ==========================================
valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
test_files = sorted([
    f.name for f in TEST_IMAGE_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in valid_exts
])

if not test_files:
    raise RuntimeError(f"No image files found in {TEST_IMAGE_DIR}")

test_df = pd.DataFrame({"Image_name": test_files})
test_df["file_path"] = test_df["Image_name"].apply(lambda x: str(TEST_IMAGE_DIR / x))

print("Total test images found:", len(test_df))

# ==========================================
# Prediction buffer (default zeros)
# If a file is bad, row stays zero and CSV still works.
# ==========================================
pred_all = np.zeros((len(test_df), len(CONDITIONS)), dtype=np.float32)
bad_files = []

# ==========================================
# Helper: load + preprocess one image
# ==========================================
def load_and_preprocess_image(path_str: str):
    raw = tf.io.read_file(path_str)  # raises if missing
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)  # jpg/png safe
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = keras.applications.resnet50.preprocess_input(img)
    return img

# ==========================================
# Batched inference with per-file error handling
# Preserves row order exactly
# ==========================================
batch_imgs = []
batch_indices = []

total = len(test_df)

for i, row in test_df.iterrows():
    path = row["file_path"]

    try:
        img = load_and_preprocess_image(path)
        batch_imgs.append(img)
        batch_indices.append(i)
    except Exception as e:
        bad_files.append((i, row["Image_name"], str(e)))
        # leave zeros in pred_all for this row
        continue

    # Run inference when batch is full
    if len(batch_imgs) == BATCH_SIZE:
        x_batch = tf.stack(batch_imgs, axis=0)
        preds = model(x_batch, training=False).numpy().astype(np.float32)
        pred_all[np.array(batch_indices)] = preds

        batch_imgs = []
        batch_indices = []

    if (i + 1) % 500 == 0:
        print(f"Processed {i+1}/{total} images...")

# Run leftover batch
if batch_imgs:
    x_batch = tf.stack(batch_imgs, axis=0)
    preds = model(x_batch, training=False).numpy().astype(np.float32)
    pred_all[np.array(batch_indices)] = preds

print("Prediction complete.")
print("Bad files skipped:", len(bad_files))

if bad_files:
    bad_report_path = BASE_DIR / "bad_test_files_report.csv"
    pd.DataFrame(bad_files, columns=["row_idx", "Image_name", "error"]).to_csv(bad_report_path, index=False)
    print("Bad test file report saved to:", bad_report_path)
    print("First 10 bad files:")
    for row in bad_files[:10]:
        print(row)

# ==========================================
# Create submission CSV with exact required columns
# ==========================================
submission = pd.DataFrame({"Image_name": test_df["Image_name"].values})

for j, col in enumerate(CONDITIONS):
    submission[col] = pred_all[:, j]

# Enforce exact column order
submission = submission[["Image_name"] + CONDITIONS]

submission_path = BASE_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

print("\nSubmission saved:", submission_path)
print("Submission shape:", submission.shape)
print(submission.head())