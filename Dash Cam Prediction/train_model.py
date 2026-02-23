import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

###############################################################################
# 1) CONFIG & DATA SPLIT
###############################################################################
data_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/processed_2frame_dataset"
all_folders = sorted(os.listdir(data_dir))

train_folders, val_folders = train_test_split(all_folders, test_size=0.2, random_state=42)
print(f"Train size: {len(train_folders)} | Val size: {len(val_folders)}")

FRAME_HEIGHT, FRAME_WIDTH = 224, 224
NUM_FRAMES = 2
BATCH_SIZE = 16  # Increased for stable learning

###############################################################################
# 2) DATA LOADING & PREPROCESSING
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

def build_dataset(folder_list, batch_size=BATCH_SIZE, shuffle=True):
    """Create a dataset"""
    ds = tf.data.Dataset.from_tensor_slices(folder_list)
    if shuffle:
        ds = ds.shuffle(len(folder_list), seed=42)
    
    ds = ds.map(tf_dataset_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

###############################################################################
# 3) TRAIN & VALIDATION DATASETS
###############################################################################
train_dataset = build_dataset(train_folders, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = build_dataset(val_folders, batch_size=BATCH_SIZE, shuffle=False)

###############################################################################
# 4) FIXED THIRD MODEL: EfficientNetB0 + Temporal CNN + Attention
###############################################################################
def build_fixed_model(input_shape=(2, 224, 224, 3)):
    """Improved Model with Fixes for Stuck Accuracy"""

    base_cnn = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_cnn.trainable = False  # Freeze CNN at start

    inputs = keras.Input(shape=input_shape)

    # ✅ Extract Features
    frame1, frame2 = inputs[:, 0], inputs[:, 1]
    feat1 = base_cnn(frame1)
    feat2 = base_cnn(frame2)

    # ✅ Global Average Pooling
    feat1 = layers.GlobalAveragePooling2D()(feat1)
    feat2 = layers.GlobalAveragePooling2D()(feat2)

    # ✅ Feature Combination
    combined_features = layers.Concatenate()([feat1, feat2])
    diff_features = layers.Subtract()([feat2, feat1])
    x = layers.Concatenate()([combined_features, diff_features])
    a = layers.Add()([diff_features, feat1])
    b = layers.Add()([diff_features, feat2])
    c = layers.Add()([a, b])  # Final sum    
    x = layers.Concatenate()([x,c])   
    # ✅ Reshape for Temporal Learning
    x = layers.Reshape((2, -1))(x)

    # ✅ Temporal CNN (Instead of ConvLSTM)
    x = layers.Conv1D(512, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(256, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(128, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(64, kernel_size=1, activation="relu")(x)
    x = layers.Conv1D(64, kernel_size=1, activation="relu")(x)
    # ✅ Improved Attention
    attn_scores = layers.Dense(1, activation="tanh")(x)
    attn_scores = layers.Softmax(axis=-1)(attn_scores)  # 🔥 Fix Softmax Axis
    context_vector = layers.Multiply()([x, attn_scores])
    x = layers.GlobalAveragePooling1D()(context_vector)

    # ✅ Stronger Dense Layers
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(1024, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(2, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, output, name="Fixed_Model")

# ✅ Compile Model with Higher Learning Rate
model = build_fixed_model()
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.AdamW(learning_rate=2e-5, weight_decay=1e-6),
    metrics=["accuracy"]
)
model.summary()

# ✅ Train Model
EPOCHS = 50
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, verbose=1)
    ]
)

print("✅ Training completed!")
model.save("third_optimized_model_fixed1.keras")

###############################################################################
# 7) EVALUATION
###############################################################################
y_true, y_pred = [], []
for frames_b, labels_b in val_dataset:
    preds = model.predict(frames_b, verbose=0)
    y_pred.extend(preds.squeeze().tolist())
    y_true.extend(labels_b.numpy().tolist())

threshold = 0.5
y_pred_binary = [1 if p >= threshold else 0 for p in y_pred]

acc = accuracy_score(y_true, y_pred_binary)
cm  = confusion_matrix(y_true, y_pred_binary)
cr  = classification_report(y_true, y_pred_binary, digits=4)

print(f"Validation Accuracy = {acc:.4f}")
print("Confusion Matrix:\n", cm)
print("Classification Report:\n", cr)
