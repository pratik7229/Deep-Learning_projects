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
data_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/processed_2frame_dataset"
all_folders = sorted(os.listdir(data_dir))

train_folders, val_folders = train_test_split(all_folders, test_size=0.3, random_state=42)
print(f"Train size: {len(train_folders)} | Val size: {len(val_folders)}")

FRAME_HEIGHT, FRAME_WIDTH = 224, 224
NUM_FRAMES = 2

###############################################################################
# 2) HELPER FUNCTIONS
###############################################################################
def parse_folder_label(folder_name):
    parts = folder_name.split("_")
    return int(parts[-1]) if len(parts) >= 2 else 0

def load_2frames(folder_tensor):
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
    frames_t, label_t = tf.py_function(load_2frames, [folder_tensor], [tf.float32, tf.float32])
    frames_t = tf.ensure_shape(frames_t, (2, FRAME_HEIGHT, FRAME_WIDTH, 3))
    label_t = tf.ensure_shape(label_t, ())
    return frames_t, label_t

def build_dataset(folder_list, batch_size=8, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices(folder_list)
    if shuffle:
        ds = ds.shuffle(len(folder_list), seed=42)
    ds = ds.map(tf_dataset_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

###############################################################################
# 3) TRAIN / VAL DATASETS
###############################################################################
train_dataset = build_dataset(train_folders, batch_size=8, shuffle=True)
val_dataset = build_dataset(val_folders, batch_size=8, shuffle=False)

###############################################################################
# 4) FIXED TRANSFORMER ENCODER BLOCK
###############################################################################
def transformer_encoder(inputs, num_heads=4, key_dim=64, ff_dim=128, dropout=0.2):
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
# 5) FIXED FINAL MODEL: MobileNetV2 + Transformer + BiLSTM + Soft Attention
###############################################################################
def build_final_model(input_shape=(2, 224, 224, 3)):
    base_cnn = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_cnn.trainable = False

    inputs = keras.Input(shape=input_shape)
    
    # ✅ Split frames properly without Lambda layers
    frame1, frame2 = inputs[:, 0], inputs[:, 1]

    # 🔥 Extract Features for Both Frames
    feat1 = base_cnn(frame1)
    feat2 = base_cnn(frame2)

    # 🔥 Global Pooling
    feat1 = layers.GlobalAveragePooling2D()(feat1)
    feat2 = layers.GlobalAveragePooling2D()(feat2)

    # 🔹 Feature Combination: CONCAT + SUBTRACTION
    combined_features = layers.Concatenate()([feat1, feat2])
    diff_features = layers.Subtract()([feat2, feat1])
    x = layers.Concatenate()([combined_features, diff_features])

    x = layers.Reshape((2, -1))(x)

    # 🔥 Transformer Block
    x = transformer_encoder(x, num_heads=8, key_dim=128, ff_dim=256, dropout=0.3)

    # 🔥 BiLSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # 🔥 Soft Attention
    attn_scores = layers.Dense(1, activation="tanh")(x)
    attn_scores = layers.Softmax(axis=1)(attn_scores)
    context_vector = layers.Multiply()([x, attn_scores])
    x = layers.GlobalAveragePooling1D()(context_vector)

    # 🔥 Fully Connected Layers
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, output, name="Final_Optimized_Model")

model = build_final_model()
model.compile(
    loss=keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy"]
)
model.summary()

###############################################################################
# 6) TRAINING
###############################################################################
def unfreeze_cnn(epoch, logs):
    if epoch == 3:
        model.get_layer("mobilenetv2_1.00_224").trainable = True
        print("\n🚀 Unfreezing CNN Backbone for fine-tuning!\n")

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

model.save("final_optimized_model1.keras")

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
