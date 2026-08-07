import cv2
import os
import sys
import argparse

def extract_frames(video_path, output_folder):
    """
    Extract frames from a video file (.avi, .mp4, etc.) into an image directory.
    """
    if not os.path.exists(video_path):
        print(f"[Error] Video file not found: {video_path}")
        sys.exit(1)

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[Error] Could not open video file: {video_path}")
        sys.exit(1)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_name = f"frame_{count:04d}.jpg"
        cv2.imwrite(os.path.join(output_folder, frame_name), frame)
        count += 1

    cap.release()
    print(f"[Success] Extracted {count} frames from '{video_path}' into '{output_folder}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract video frames into an image folder")
    parser.add_argument("video_path", help="Path to input video file (e.g. video.avi)")
    parser.add_argument("output_folder", help="Target output folder for frames")
    
    args = parser.parse_args()
    extract_frames(args.video_path, args.output_folder)
