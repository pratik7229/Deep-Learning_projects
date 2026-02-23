import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

###############################################################################
# 1) Configuration
###############################################################################
data_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/processed_unlabeled"
all_folders = sorted(os.listdir(data_dir))  # subfolders for unlabeled/unseen data
print(f"Number of unseen folders: {len(all_folders)}")

model_path = "fast_model1.keras"  # Path to your trained model

###############################################################################
# 2) Helper Functions
###############################################################################

def resize_video_frames(frames: np.ndarray) -> np.ndarray:
    """
    Resize from (num_frames,H,W,3) to (num_frames,128,128,3).
    """
    num_frames, old_h, old_w, ch = frames.shape
    out = np.zeros((num_frames, 128, 128, ch), dtype=np.float32)
    for i in range(num_frames):
        out[i] = cv2.resize(frames[i], (128, 128)).astype(np.float32)
    return out

def load_data_pyfunc(folder_tensor):
    """
    Python function for tf.py_function to load frames, alert, event 
    from a subfolder. We IGNORE any label info because data is unlabeled.
    """
    folder_str = folder_tensor.numpy().decode("utf-8")  # Convert tf.string -> python str
    folder_path = os.path.join(data_dir, folder_str)

    # 1) Load frames
    frames_path = os.path.join(folder_path, "frames.npy")
    if os.path.exists(frames_path):
        frames = np.load(frames_path).astype(np.float32)
        frames = resize_video_frames(frames)
        frames /= 255.0
    else:
        frames = np.zeros((30,128,128,3), dtype=np.float32)

    # 2) Load alert & event images
    def load_image(path):
        if os.path.exists(path):
            img = cv2.imread(path).astype(np.float32)
            img = cv2.resize(img, (128,128)) / 255.0
        else:
            img = np.zeros((128,128,3), dtype=np.float32)
        return img

    alert_frame_path = os.path.join(folder_path, "alert_frame.jpg")
    alert_frame = load_image(alert_frame_path)

    event_frame_path = os.path.join(folder_path, "event_frame.jpg")
    event_frame = load_image(event_frame_path)

    # We have NO label for this unseen data. Return a dummy label (e.g., -1)
    # or just skip it. We'll return -1 so the pipeline remains consistent.
    dummy_label = -1.0

    return frames, alert_frame, event_frame, dummy_label

def tf_load_data(folder_tensor):
    """
    tf.py_function wrapper:
      Returns => (frames, alert, event) , dummy_label
    """
    frames_t, alert_t, event_t, label_t = tf.py_function(
        load_data_pyfunc,
        [folder_tensor],
        [tf.float32, tf.float32, tf.float32, tf.float32]
    )

    frames_t = tf.ensure_shape(frames_t, (30,128,128,3))
    alert_t  = tf.ensure_shape(alert_t,  (128,128,3))
    event_t  = tf.ensure_shape(event_t,  (128,128,3))
    label_t  = tf.ensure_shape(label_t,  ())

    # Return final structure => no real label, just dummy_label
    return (frames_t, alert_t, event_t), label_t

def build_inference_dataset(folder_list, batch_size=8):
    ds = tf.data.Dataset.from_tensor_slices(folder_list)
    ds = ds.map(tf_load_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

###############################################################################
# 3) Build Inference Dataset
###############################################################################
inference_dataset = build_inference_dataset(all_folders, batch_size=8)

###############################################################################
# 4) Load the Trained Model
###############################################################################
model = keras.models.load_model(model_path)
model.summary()

###############################################################################
# 5) Predict on Unseen Data
###############################################################################
predictions = []  # store (folder_str, probability)

folder_index = 0
for (frames_b, alert_b, event_b), dummy_label_b in inference_dataset:
    # model.predict returns probabilities (sigmoid) for each sample
    probs = model.predict([frames_b, alert_b, event_b], verbose=0)

    # For each sample in this batch, store it
    batch_size = probs.shape[0]
    for i in range(batch_size):
        folder_str = all_folders[folder_index + i]  # track which folder
        pred_prob  = probs[i,0]  # the sigmoid probability
        predictions.append((folder_str, float(pred_prob)))
    folder_index += batch_size

###############################################################################
# 6) Print or Save Predictions
###############################################################################
# Example: Print the first 10
print("Sample predictions on unseen data (folder -> probability):")
for i in range(min(10, len(predictions))):
    print(predictions[i])

# Optionally, save predictions to CSV
import csv
output_csv = "unlabeled_predictions.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["folder", "predicted_probability"])
    for folder_str, prob in predictions:
        writer.writerow([folder_str, prob])

print(f"✅ Inference done. Saved predictions to {output_csv}")


import csv

# Create submission CSV
with open("submission.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "target"])  # header

    for folder_str, prob in predictions:
        # Extract just the video ID. 
        # If folder_str is something like "00204_0", 
        # we might split on underscore:
        video_id = folder_str.split("_")[0]
        
        # Convert probability to 0 or 1
        target = 1 if prob >= 0.5 else 0
        
        writer.writerow([video_id, target])

print("✅ submission.csv created with columns [id, target].")
