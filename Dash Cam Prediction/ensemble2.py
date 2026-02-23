import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

###############################################################################
# 1) CONFIGURATION & DATA LOADING
###############################################################################
data_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/processed_2frame_dataset"
all_folders = sorted(os.listdir(data_dir))

train_folders, val_folders = train_test_split(all_folders, test_size=0.3, random_state=42)
print(f"Train size: {len(train_folders)} | Val size: {len(val_folders)}")

FRAME_HEIGHT, FRAME_WIDTH = 224, 224
NUM_FRAMES = 2
BATCH_SIZE = 8

###############################################################################
# 2) DATA AUGMENTATION
###############################################################################
augment_layer = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1, 0.1),
    layers.RandomContrast(0.1),
    layers.GaussianNoise(0.05),
    layers.RandomBrightness(0.2)
], name="augmentation")

def augment_two_frames(frames, label):
    """Applies augmentation to both frames"""
    return tf.stack([augment_layer(frames[0]), augment_layer(frames[1])], axis=0), label

###############################################################################
# 3) DATA LOADING & PREPROCESSING
###############################################################################
def parse_folder_label(folder_name):
    """Extract label (0 or 1) from folder name"""
    parts = folder_name.split("_")
    return int(parts[-1]) if len(parts) >= 2 else 0

def load_2frames(folder_tensor):
    """Load 2 consecutive frames and their label"""
    folder_str = folder_tensor.numpy().decode("utf-8")
    folder_path = os.path.join(data_dir, folder_str)
    label = parse_folder_label(folder_str)

    frames_path = os.path.join(folder_path, "frames.npy")
    if os.path.exists(frames_path):
        frames = np.load(frames_path).astype(np.float32)
    else:
        frames = np.zeros((2, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)

    frames = np.clip(frames / 255.0, 0.0, 1.0)  # Normalize
    return frames, np.float32(label)

def tf_dataset_map(folder_tensor):
    """Wrap `load_2frames` inside a TensorFlow function"""
    frames_t, label_t = tf.py_function(load_2frames, [folder_tensor], [tf.float32, tf.float32])
    frames_t = tf.ensure_shape(frames_t, (2, FRAME_HEIGHT, FRAME_WIDTH, 3))
    label_t = tf.ensure_shape(label_t, ())
    return frames_t, label_t

def build_dataset(folder_list, batch_size=BATCH_SIZE, shuffle=True, augment=False):
    """Create a dataset with optional augmentation"""
    ds = tf.data.Dataset.from_tensor_slices(folder_list)
    if shuffle:
        ds = ds.shuffle(len(folder_list), seed=42)
    
    ds = ds.map(tf_dataset_map, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        ds = ds.map(augment_two_frames, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

###############################################################################
# 4) TRAIN & VALIDATION DATASETS
###############################################################################
train_dataset = build_dataset(train_folders, batch_size=BATCH_SIZE, shuffle=True, augment=True)
val_dataset = build_dataset(val_folders, batch_size=BATCH_SIZE, shuffle=False, augment=False)

###############################################################################
# 5) ENSEMBLE MODEL WITH TRAINABLE FUSION LAYER
###############################################################################
def build_ensemble_model(model_A, model_B, model_C):
    """Combines three models using Trainable Fusion & Transformer Attention."""
    
    input_shape = (2, 224, 224, 3)  # Adjust based on frame size
    inputs = keras.Input(shape=input_shape)

    # ✅ Ensure unique names for each model instance
    model_A = keras.Model(model_A.input, model_A.output, name="Model_A")
    model_B = keras.Model(model_B.input, model_B.output, name="Model_B")
    model_C = keras.Model(model_C.input, model_C.output, name="Model_C")

    # Get model predictions
    preds_A = model_A(inputs)
    preds_B = model_B(inputs)
    preds_C = model_C(inputs)

    # Ensure predictions have the same shape
    preds_A = layers.Reshape((1,), name="Reshape_A")(preds_A)
    preds_B = layers.Reshape((1,), name="Reshape_B")(preds_B)
    preds_C = layers.Reshape((1,), name="Reshape_C")(preds_C)

    # ✅ Trainable Fusion Layer (Dense + Softmax Weights)
    stacked_preds = layers.Concatenate(name="Concat_Features")([preds_A, preds_B, preds_C])
    fusion_weights = layers.Dense(3, activation="softmax", name="Fusion_Weights")(stacked_preds)
    weighted_sum = layers.Dot(axes=1)([fusion_weights, stacked_preds])  # Weighted Sum

    # ✅ Fully Connected Layers
    x = layers.BatchNormalization()(weighted_sum)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid", name="Final_Output")(x)  # Final probability

    return keras.Model(inputs, x, name="Trainable_Fusion_Ensemble")

###############################################################################
# 6) LOAD PRE-TRAINED MODELS
###############################################################################
print("\n🚀 Loading Pre-Trained Models...\n")
model_A = keras.models.load_model("/Users/pratik/Documents/Projects/dashcam prediction/final_optimized_model1.keras")  
model_B = keras.models.load_model("/Users/pratik/Documents/Projects/dashcam prediction/final_optimized_model.keras")  
model_C = keras.models.load_model("/Users/pratik/Documents/Projects/dashcam prediction/third_optimized_model_fixed1.keras")  

# Freeze models
model_A.trainable = False
model_B.trainable = False
model_C.trainable = False

ensemble_model = build_ensemble_model(model_A, model_B, model_C)

ensemble_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    metrics=["accuracy"]
)

ensemble_model.summary()

###############################################################################
# 7) TRAINING
###############################################################################
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

EPOCHS = 15
history = ensemble_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

ensemble_model.save("trainable_fusion_ensemble.keras")

print("✅ Training completed!")

###############################################################################
# 8) EVALUATION
###############################################################################
y_true, y_pred = [], []
for frames_b, labels_b in val_dataset:
    preds = ensemble_model.predict(frames_b, verbose=0)
    y_pred.extend(preds.squeeze().tolist())
    y_true.extend(labels_b.numpy().tolist())

# Convert probabilities -> binary labels
y_pred_binary = [1 if p >= 0.5 else 0 for p in y_pred]

# Compute Metrics
acc = accuracy_score(y_true, y_pred_binary)
cm  = confusion_matrix(y_true, y_pred_binary)
cr  = classification_report(y_true, y_pred_binary, digits=4)

print(f"Validation Accuracy = {acc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)
