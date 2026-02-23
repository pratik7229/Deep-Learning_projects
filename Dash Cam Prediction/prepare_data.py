import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Paths
test_videos_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/test"
output_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/processed_test_20frame_dataset"
os.makedirs(output_dir, exist_ok=True)

# Constants
FRAME_HEIGHT, FRAME_WIDTH = 224, 224
NUM_FRAMES = 20  # Extract 20 consecutive frames
black_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

def extract_20_consecutive_frames(video_path):
    """
    Extracts 20 consecutive frames from the middle of the video.
    If the video has fewer than 20 frames, it is skipped.
    Returns a NumPy array of shape (20, 224, 224, 3).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < NUM_FRAMES:
        cap.release()
        return None  # Skip videos with fewer than 20 frames

    # Start extracting from the middle of the video
    start_idx = (total_frames - NUM_FRAMES) // 2
    selected_indices = list(range(start_idx, start_idx + NUM_FRAMES))  # Consecutive 20 frames

    frames = []
    for idx in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frames.append(black_frame.copy())  # Use black frame if reading fails
        else:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frames.append(frame)

    cap.release()
    
    return np.array(frames, dtype=np.uint8)  # Shape: (20, 224, 224, 3)

def main():
    video_files = sorted(os.listdir(test_videos_dir))

    for video_file in tqdm(video_files):
        video_id = os.path.splitext(video_file)[0]  # Extract video ID from filename
        video_path = os.path.join(test_videos_dir, video_file)

        frames_20 = extract_20_consecutive_frames(video_path)
        if frames_20 is None:
            print(f"⚠️ Skipping {video_id}: Not enough frames.")
            continue

        # Save frames as "<video_id>.npy"
        output_path = os.path.join(output_dir, f"{video_id}.npy")
        np.save(output_path, frames_20)

    print("\n✅ Done creating 20-frame test dataset! Files saved in:", output_dir)

if __name__ == "__main__":
    main()
