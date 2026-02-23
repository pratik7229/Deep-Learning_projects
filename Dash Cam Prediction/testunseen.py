import os
import cv2
import numpy as np
from tqdm import tqdm

# 1) PATHS
unlabeled_videos_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/train"  # Directory containing *unseen* .mp4 files
output_dir = "/Users/pratik/Documents/Projects/dashcam prediction/nexar-collision-prediction/processed_unlabeled"       # Output directory for extracted data
os.makedirs(output_dir, exist_ok=True)

# 2) CONSTANTS
NUM_FRAMES = 30
FRAME_HEIGHT, FRAME_WIDTH = 224, 224
black_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

# 3) Function to process each video
def extract_fixed_length_frames_unlabeled(video_path, video_id):
    """
    Extract 30 frames from the .mp4 by sampling every 1 second,
    plus create black placeholders for alert & event frames,
    and store dummy metadata of [-1, -1, -1].
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    # We'll store black placeholders
    extracted_alert_frame = black_frame.copy()
    extracted_event_frame = black_frame.copy()

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for consistency
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Grab 1 frame per second
        if i % fps == 0:
            frames.append(frame)

    cap.release()

    # Convert frames to NumPy
    frames = np.array(frames)  # shape: (num_frames, H, W, C)

    # Ensure exactly NUM_FRAMES
    if len(frames) > NUM_FRAMES:
        # Sample evenly to get 30
        indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
        frames = frames[indices]
    elif len(frames) < NUM_FRAMES:
        # Pad with black frames
        pad_count = NUM_FRAMES - len(frames)
        pad_array = np.array([black_frame] * pad_count)
        frames = np.vstack((frames, pad_array))

    # Create output folder for this video
    video_output_folder = os.path.join(output_dir, video_id)
    os.makedirs(video_output_folder, exist_ok=True)

    # Save frames.npy
    np.save(os.path.join(video_output_folder, "frames.npy"), frames)

    # Save black alert/event frames
    cv2.imwrite(os.path.join(video_output_folder, "alert_frame.jpg"), extracted_alert_frame)
    cv2.imwrite(os.path.join(video_output_folder, "event_frame.jpg"), extracted_event_frame)

    # Save dummy metadata => [-1, -1, -1]
    np.save(os.path.join(video_output_folder, "metadata.npy"), np.array([-1, -1, -1]))

# 4) MAIN: Loop over all .mp4 files in unlabeled_videos_dir
def main():
    # Gather all .mp4 files
    all_videos = [f for f in os.listdir(unlabeled_videos_dir) if f.endswith(".mp4")]

    for video_file in tqdm(all_videos, desc="Processing unlabeled videos"):
        video_id = os.path.splitext(video_file)[0]  # remove .mp4
        video_path = os.path.join(unlabeled_videos_dir, video_file)

        extract_fixed_length_frames_unlabeled(video_path, video_id)

    print("✅ Finished preparing unlabeled data.")

if __name__ == "__main__":
    main()
