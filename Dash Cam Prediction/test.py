import os
import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tqdm import tqdm
import keras

###############################################################################
# 1) CONFIGURATION & PATHS
###############################################################################
test_videos_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/test"  # Folder containing test videos
output_csv_path = "/Users/pratik/Documents/Projects/dashcam prediction/submission_raw_predictionsnew.csv"

FRAME_HEIGHT, FRAME_WIDTH = 224, 224
NUM_FRAMES = 20  # Extract 10 frames per video

# Load trained ensemble model
print("\n🚀 Loading Trained Ensemble Model...\n")
keras.config.enable_unsafe_deserialization()
ensemble_model = keras.models.load_model("/Users/pratik/Documents/Projects/dashcam prediction/trainable_fusion_ensemble.keras")

###############################################################################
# 2) VIDEO FRAME EXTRACTION FUNCTION
###############################################################################
def extract_frames_from_video(video_path, num_frames=NUM_FRAMES):
    """
    Extracts `num_frames` evenly spaced frames from a video.
    If the video has less than `num_frames`, frames are duplicated.
    Returns a NumPy array of shape (num_frames, H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 2:
        cap.release()
        return None  # Skip videos with less than 2 frames

    # Select `num_frames` evenly spaced indices
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = frame.astype(np.float32) / 255.0  # Normalize to [0,1]
        frames.append(frame)

    cap.release()

    if len(frames) < num_frames:
        return None  # Skip if extraction fails

    return np.array(frames)  # Shape (num_frames, H, W, 3)

###############################################################################
# 3) INFERENCE FUNCTION (RAW PROBABILITIES)
###############################################################################
def predict_video(video_path):
    """
    Given a video file, extract 10 frames, predict on 2-frame pairs,
    and return the average probability as the final prediction.
    """
    frames = extract_frames_from_video(video_path)
    if frames is None:
        return None  # Skip this video

    raw_predictions = []

    # Pass pairs of consecutive frames into the model
    for i in range(NUM_FRAMES - 1):  # Pairs: (frame_0, frame_1), (frame_1, frame_2), ...
        frame_pair = np.array([frames[i], frames[i + 1]])  # Shape (2, H, W, 3)
        frame_pair = np.expand_dims(frame_pair, axis=0)  # Add batch dimension -> (1, 2, H, W, 3)

        # Predict using the trained model
        prob = ensemble_model.predict(frame_pair, verbose=0)[0][0]  # Single probability output
        raw_predictions.append(prob)

    # Compute the **average raw probability** across all frame pairs
    avg_prediction = np.mean(raw_predictions)

    # Return probability with **5 decimal places**
    return round(avg_prediction, 5)

###############################################################################
# 4) PROCESS TEST VIDEOS & SAVE OUTPUT
###############################################################################
submission_data = []

print("\n🚀 Running Inference on Test Videos...\n")
for video_file in tqdm(os.listdir(test_videos_dir)):
    video_path = os.path.join(test_videos_dir, video_file)

    # Extract Video ID (Assumes filename format "00123.mp4" -> Extract "00123")
    video_id = os.path.splitext(video_file)[0]

    # Run prediction
    prediction = predict_video(video_path)
    if prediction is not None:
        submission_data.append({"id": video_id, "target": prediction})

# Convert results to DataFrame and save as CSV
submission_df = pd.DataFrame(submission_data)

# Ensure the target column has 5 decimal places
submission_df["target"] = submission_df["target"].apply(lambda x: f"{x:.5f}")

submission_df.to_csv(output_csv_path, index=False)

print(f"\n✅ Submission file saved: {output_csv_path}")
print(submission_df.head())  # Show first few predictions
