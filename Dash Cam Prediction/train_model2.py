import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

###############################################################################
# 1) CONFIG & DATA SPLIT
###############################################################################
data_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/processed_4frame_dataset"
all_files = sorted(os.listdir(data_dir))

train_files, val_files = train_test_split(all_files, test_size=0.3, random_state=42)
print(f"Train size: {len(train_files)} | Val size: {len(val_files)}")

FRAME_HEIGHT, FRAME_WIDTH = 224, 224
NUM_FRAMES = 4  # Using 4 frames per sample
BATCH_SIZE = 8

###############################################################################
# 2) FIXED AUGMENTATION PIPELINE
###############################################################################
augment_layer = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1, 0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

def augment_four_frames(frames, label):
    """Applies augmentation to all four frames."""
    return tf.stack([augment_layer(frames[i]) for i in range(NUM_FRAMES)], axis=0), label

###############################################################################
# 3) DATA LOADING & PREPROCESSING
###############################################################################
def parse_filename_label(filename):
    """Extracts label from filename '<video_id>_<target>.npy'."""
    parts = filename.split("_")
    return int(parts[-1].split(".")[0]) if len(parts) >= 2 else 0

def load_4frames(filename_tensor):
    """Loads .npy files and returns (frames, label)."""
    filename_str = filename_tensor.numpy().decode("utf-8")
    file_path = os.path.join(data_dir, filename_str)
    label = parse_filename_label(filename_str)

    if os.path.exists(file_path):
        frames = np.load(file_path).astype(np.float32)
    else:
        frames = np.zeros((NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)

    frames = np.clip(frames / 255.0, 0.0, 1.0)
    return frames, np.float32(label)

def tf_dataset_map(filename_tensor):
    """Wrap `load_4frames` inside a TensorFlow function"""
    frames_t, label_t = tf.py_function(load_4frames, [filename_tensor], [tf.float32, tf.float32])
    frames_t = tf.ensure_shape(frames_t, (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3))
    label_t = tf.ensure_shape(label_t, ())
    return frames_t, label_t

def build_dataset(file_list, batch_size=BATCH_SIZE, shuffle=True, augment=False):
    """Create a dataset with optional augmentation"""
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle:
        ds = ds.shuffle(len(file_list), seed=42)
    
    ds = ds.map(tf_dataset_map, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        ds = ds.map(augment_four_frames, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

###############################################################################
# 4) TRAIN & VALIDATION DATASETS
###############################################################################
train_dataset = build_dataset(train_files, batch_size=BATCH_SIZE, shuffle=True, augment=True)
val_dataset = build_dataset(val_files, batch_size=BATCH_SIZE, shuffle=False, augment=False)

###############################################################################
# 5) IMPROVED TRANSFORMER ENCODER
###############################################################################
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

###############################################################################
# 1) CONFIG & DATA SPLIT
###############################################################################
data_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/processed_4frame_dataset"
all_files = sorted(os.listdir(data_dir))

train_files, val_files = train_test_split(all_files, test_size=0.3, random_state=42)
print(f"Train size: {len(train_files)} | Val size: {len(val_files)}")

FRAME_HEIGHT, FRAME_WIDTH = 224, 224
NUM_FRAMES = 4  # Using 4 frames per sample
BATCH_SIZE = 8

###############################################################################
# 2) AUGMENTATION PIPELINE
###############################################################################
augment_layer = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1, 0.1),
    layers.RandomContrast(0.1),
], name="augmentation")

def augment_four_frames(frames, label):
    """Applies augmentation to all four frames."""
    return tf.stack([augment_layer(frames[i]) for i in range(NUM_FRAMES)], axis=0), label

###############################################################################
# 3) DATA LOADING & PREPROCESSING
###############################################################################
def parse_filename_label(filename):
    """Extracts label from filename '<video_id>_<target>.npy'."""
    parts = filename.split("_")
    return int(parts[-1].split(".")[0]) if len(parts) >= 2 else 0

def load_4frames(filename_tensor):
    """Loads .npy files and returns (frames, label)."""
    filename_str = filename_tensor.numpy().decode("utf-8")
    file_path = os.path.join(data_dir, filename_str)
    label = parse_filename_label(filename_str)

    if os.path.exists(file_path):
        frames = np.load(file_path).astype(np.float32)
    else:
        frames = np.zeros((NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)

    frames = np.clip(frames / 255.0, 0.0, 1.0)
    return frames, np.float32(label)

def tf_dataset_map(filename_tensor):
    """Wrap `load_4frames` inside a TensorFlow function"""
    frames_t, label_t = tf.py_function(load_4frames, [filename_tensor], [tf.float32, tf.float32])
    frames_t = tf.ensure_shape(frames_t, (NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3))
    label_t = tf.ensure_shape(label_t, ())
    return frames_t, label_t

def build_dataset(file_list, batch_size=BATCH_SIZE, shuffle=True, augment=False):
    """Create a dataset with optional augmentation"""
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    if shuffle:
        ds = ds.shuffle(len(file_list), seed=42)
    
    ds = ds.map(tf_dataset_map, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        ds = ds.map(augment_four_frames, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

###############################################################################
# 4) TRAIN & VALIDATION DATASETS
###############################################################################
train_dataset = build_dataset(train_files, batch_size=BATCH_SIZE, shuffle=True, augment=True)
val_dataset = build_dataset(val_files, batch_size=BATCH_SIZE, shuffle=False, augment=False)

###############################################################################
# 5) IMPROVED TRANSFORMER ENCODER
###############################################################################
def transformer_encoder(inputs, num_heads=8, key_dim=128, ff_dim=256, dropout=0.3):
    """Improved Transformer Encoder Block"""
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = layers.Add()([x, attn_output])

    x_ff = layers.Dense(ff_dim, activation="relu")(x)
    x_ff = layers.Dense(inputs.shape[-1])(x_ff)
    x = layers.Add()([x, x_ff])
    x = layers.Dropout(dropout)(x)

    return x

###############################################################################
# 6) FINAL MODEL: EfficientNetV2 + Transformer + BiLSTM + Soft Attention
###############################################################################
def build_final_model(input_shape=(4, 224, 224, 3)):
    """Optimized architecture combining CNN, Transformer, BiLSTM, and Attention"""
    
    base_cnn = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_cnn.trainable = False  # Freeze initial layers

    inputs = keras.Input(shape=input_shape)

    frame_features = []
    for i in range(NUM_FRAMES):
        frame = layers.Lambda(lambda x, idx=i: x[:, idx], name=f"frame_{i}")(inputs)  # Extract frame `i`
        feat = base_cnn(frame)
        feat = layers.GlobalAveragePooling2D()(feat)  # Convert to (batch, feature_dim)
        frame_features.append(feat)

    # 🔹 Feature Combination: Concatenation & Subtraction
    concat_features = layers.Concatenate()(frame_features)
    diff_features = []
    for i in range(NUM_FRAMES - 1):
        diff = layers.Subtract()([frame_features[i + 1], frame_features[i]])
        diff_features.append(diff)
    diff_features = layers.Concatenate()(diff_features)

    # Combine concatenated and subtracted features
    x = layers.Concatenate()([concat_features, diff_features])

    # 🔥 Transformer Block
    x = layers.Reshape((NUM_FRAMES - 1 + NUM_FRAMES, -1))(x)
    x = transformer_encoder(x)

    # 🔥 BiLSTM
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # 🔥 Soft Attention
    attn_scores = layers.Dense(1, activation="tanh")(x)
    attn_scores = layers.Softmax(axis=1)(attn_scores)
    context_vector = layers.Multiply()([x, attn_scores])
    x = layers.GlobalAveragePooling1D()(context_vector)

    # 🔥 Fully Connected Layers
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    
    output = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, output, name="Final_Optimized_Model")

model = build_final_model()
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
    metrics=["accuracy"]
)
model.summary()


###############################################################################
# 7) TRAINING
###############################################################################
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

EPOCHS = 30
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)
print("✅ Training completed!")

model.save("final_optimized_4frame_model1.keras")
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