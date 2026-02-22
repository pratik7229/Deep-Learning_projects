import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
print("GPU Devices:", tf.config.list_physical_devices("GPU"))

BASE_DIR = Path("/Users/pratik/Documents/Projects/kaggleCompetation data new/grand-xray-slam-division-b")
IMAGE_DIR = BASE_DIR / "train2"
CSV_PATH = BASE_DIR / "train2.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# -----------------------------
# Load CSV (NO full decode scan)
# -----------------------------
image_df = pd.read_csv(CSV_PATH)

# Clean filenames only
image_df = image_df.dropna(subset=["Image_name"]).copy()
image_df["Image_name"] = image_df["Image_name"].astype(str).str.strip()
image_df["file_path"] = image_df["Image_name"].apply(lambda x: str(IMAGE_DIR / x))

# Optional fast filter (cheap): remove obvious missing files only
# This is much faster than decoding every image
image_df = image_df[image_df["file_path"].apply(os.path.exists)].reset_index(drop=True)

# Split
train_df, val_df = train_test_split(image_df, test_size=0.2, random_state=42, shuffle=True)

train_paths = train_df["file_path"].values
train_labels = train_df[CONDITIONS].values.astype("float32")

val_paths = val_df["file_path"].values
val_labels = val_df[CONDITIONS].values.astype("float32")

print("Training paths shape:", train_paths.shape)
print("Training labels shape:", train_labels.shape)
print("Validation paths shape:", val_paths.shape)
print("Validation labels shape:", val_labels.shape)

# -----------------------------
# Robust parser (handles jpg/png, skips bad files via ignore_errors)
# -----------------------------
def parse_image(filepath, label):
    raw_data = tf.io.read_file(filepath)  # fails if missing -> ignore_errors will skip
    image = tf.io.decode_image(raw_data, channels=3, expand_animations=False)  # robust decoder
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    # ResNet50 preprocessing
    image = keras.applications.resnet50.preprocess_input(image)

    return image, label

def build_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=min(4096, len(paths)), reshuffle_each_iteration=True)

    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)

    # KEY LINE: skip bad files during training instead of crashing
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = build_dataset(train_paths, train_labels, training=True)
val_ds = build_dataset(val_paths, val_labels, training=False)

# -----------------------------
# Model
# -----------------------------
inputs = keras.Input(shape=(224, 224, 3))
base_model = keras.applications.ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
base_model.trainable = False  # faster first pass

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(len(CONDITIONS), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc")]
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(str(BASE_DIR / "best_xray_model.keras"), monitor="val_loss", save_best_only=True)
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)