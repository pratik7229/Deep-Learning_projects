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

###############################################################################
# 2) HELPER FUNCTIONS (DATA LOADING)
###############################################################################
def parse_folder_label(folder_name):
    """Extracts target label from folder name '<video_id>_<target>'."""
    parts = folder_name.split("_")
    return int(parts[-1]) if len(parts) >= 2 else 0

def load_2frames(folder_tensor):
    """Loads frames.npy and returns (frames, label)."""
    folder_str = folder_tensor.numpy().decode("utf-8")
    folder_path = os.path.join(data_dir, folder_str)
    label = parse_folder_label(folder_str)

    frames_path = os.path.join(folder_path, "frames.npy")
    if os.path.exists(frames_path):
        frames = np.load(frames_path).astype(np.float32)
    else:
        frames = np.zeros((2, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)

    frames = np.clip(frames / 255.0, 0.0, 1.0)
    return frames, np.float32(label)

def tf_dataset_map(folder_tensor):
    """Wraps load_2frames with tf.py_function."""
    frames_t, label_t = tf.py_function(load_2frames, [folder_tensor], [tf.float32, tf.float32])
    frames_t = tf.ensure_shape(frames_t, (2, FRAME_HEIGHT, FRAME_WIDTH, 3))
    label_t = tf.ensure_shape(label_t, ())
    return frames_t, label_t

def build_dataset(folder_list, batch_size=4, shuffle=True):
    """Builds TF dataset pipeline."""
    ds = tf.data.Dataset.from_tensor_slices(folder_list)
    if shuffle:
        ds = ds.shuffle(len(folder_list), seed=42)
    ds = ds.map(tf_dataset_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

###############################################################################
# 3) TRAIN / VAL DATASETS
###############################################################################
train_dataset = build_dataset(train_folders, batch_size=4, shuffle=True)
val_dataset = build_dataset(val_folders, batch_size=4, shuffle=False)

###############################################################################
# 4) LOAD PRE-TRAINED MODELS
###############################################################################
print("\n🚀 Loading Pre-Trained Models...\n")
model_A = keras.models.load_model("/Users/pratik/Documents/Projects/dashcam prediction/final_optimized_model1.keras")  # 97% Train Accuracy
model_B = keras.models.load_model("/Users/pratik/Documents/Projects/dashcam prediction/final_optimized_model.keras")  # 84% Validation Accuracy

# Ensure both models output logits, not probabilities
model_A = keras.Model(model_A.input, model_A.output, name="Model_A")
model_B = keras.Model(model_B.input, model_B.output, name="Model_B")

# Freeze both models so they don't change during training
model_A.trainable = False
model_B.trainable = False

###############################################################################
# 5) BUILD ENSEMBLE MODEL (ENHANCED)
###############################################################################
def build_ensemble_model(model_A, model_B):
    """Combines two models and learns optimal weightage with additional dense layers."""
    
    input_shape = (2, FRAME_HEIGHT, FRAME_WIDTH, 3)
    inputs = keras.Input(shape=input_shape)

    # Get raw predictions (logits before activation)
    logits_A = model_A(inputs)
    logits_B = model_B(inputs)

    # Ensure they have the same shape
    logits_A = layers.Reshape((1,))(logits_A)
    logits_B = layers.Reshape((1,))(logits_B)

    # Stack logits for weighting
    combined_logits = layers.Concatenate()([logits_A, logits_B])  # Shape (batch, 2)

    # Learn optimal weightage via trainable Softmax
    weight_scores = layers.Dense(2, activation="softmax")(combined_logits)  # Softmax weights

    # Compute final prediction as weighted sum
    weighted_sum = layers.Dot(axes=1)([weight_scores, combined_logits])  # Weighted Sum

    # 🔥 Additional Learning Layers 🔥
    x = layers.Dense(128, activation="relu")(weighted_sum)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)  # Final probability

    return keras.Model(inputs, x, name="Enhanced_Ensemble_Model")

ensemble_model = build_ensemble_model(model_A, model_B)

ensemble_model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4),
    metrics=["accuracy"]
)

ensemble_model.summary()

###############################################################################
# 6) TRAIN ENSEMBLE MODEL
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

print("✅ Training completed!")

# Save ensemble model
ensemble_model.save("enhanced_ensemble_model.keras")

###############################################################################
# 7) EVALUATE THE ENSEMBLE
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
