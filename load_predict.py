import argparse
import os
import sys
import time
import cv2
import numpy as np
import joblib

def preprocess_frame(frame):
    """
    Preprocess frame to match the trained MDT model dimensions (17x14 = 238 pixels * 2 flow components = 476 features).
    Keep frame in uint8 [0, 255] range for accurate Farneback optical flow calculation.
    """
    if frame is None:
        raise ValueError("Invalid frame: Received None.")
    
    # Resize frame to (17, 14) matching model features
    resized_frame = cv2.resize(frame, (17, 14))

    return resized_frame

def extract_features(frames):
    """
    Extract optical flow features between consecutive frames using Farneback optical flow.
    """
    features = []

    if len(frames) < 2:
        print("[Warning] Need at least 2 frames to calculate optical flow.")
        return np.array(features)

    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]

        # Calculate Farneback optical flow
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Flatten flow matrix (14, 17, 2) -> (476,)
        flow_flat = flow.reshape(-1)
        features.append(flow_flat)

    return np.array(features)

def compute_anomaly_scores(mdt_model, features):
    """
    Compute continuous anomaly scores using GMM negative log-likelihood (-score_samples).
    Higher scores indicate higher likelihood of abnormal behavior.
    """
    # GMM score_samples returns log-likelihood. Negating gives anomaly score.
    log_likelihood = mdt_model.score_samples(features)
    anomaly_scores = -log_likelihood
    return anomaly_scores

def further_processing(anomaly_scores, threshold=None):
    """
    Apply thresholding to continuous anomaly scores to produce binary predictions (0: Normal, 1: Anomaly).
    If threshold is None, adaptively threshold at mean + 0.3 * std of clip scores.
    """
    if threshold is None:
        threshold = np.mean(anomaly_scores) + 0.3 * np.std(anomaly_scores)
    processed_predictions = (anomaly_scores > threshold).astype(int)
    return processed_predictions, threshold


def print_score_summary(anomaly_scores, processed_predictions, threshold_used, start_frame=1, end_frame=None):
    """
    Print per-transition anomaly score details for a selected frame window.
    Frame numbers are 1-based and refer to the original input frame sequence.
    """
    if end_frame is None:
        end_frame = start_frame + len(anomaly_scores)

    print("\n[Score Detail] Selected frame window analysis:")
    print(f"  -> Window frames: {start_frame} to {end_frame}")
    print(f"  -> Threshold used: {threshold_used:.2f}")

    for idx, score in enumerate(anomaly_scores):
        transition_start = start_frame + idx
        transition_end = transition_start + 1
        label = "ANOMALY" if processed_predictions[idx] == 1 else "NORMAL"
        print(f"  -> Frame {transition_start} -> {transition_end}: score={score:.2f}, label={label}")

    print(f"  -> Max score: {np.max(anomaly_scores):.2f}")
    print(f"  -> Mean score: {np.mean(anomaly_scores):.2f}")


def main():
    print("=" * 60)
    print(" CROWD ANOMALY DETECTION INFERENCE PIPELINE (STEP-BY-STEP)")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Crowd anomaly detection inference")
    parser.add_argument("frames_directory", nargs="?", default=None,
                        help="Path to the folder containing input frames")
    parser.add_argument("--show", action="store_true", help="Display the preview window")
    parser.add_argument("--popup", action="store_true", help="Show anomaly popup windows")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Start frame index (1-based) for detailed anomaly score analysis")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="End frame index (1-based) for detailed anomaly score analysis")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="FPS speed for the output video playback (default: 30.0)")
    args = parser.parse_args()

    # STEP 1: Path & Model Resolution
    print("\n[Step 1/5] Loading trained MDT (GMM) model...")
    t0 = time.time()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'model.pkl')
    
    if args.frames_directory is not None:
        frames_directory = args.frames_directory
    else:
        frames_directory = os.path.join(
            base_dir, 'dataset', 'UCSD_Anomaly_Dataset.v1p2', 'UCSDped1', 'Test', 'Test001'
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(frames_directory):
        raise FileNotFoundError(f"Frames directory not found at: {frames_directory}")

    mdt_model = joblib.load(model_path)
    print(f"  -> Loaded model successfully from '{os.path.basename(model_path)}' in {time.time()-t0:.2f}s")
    print(f"  -> Model components: {getattr(mdt_model, 'n_components', 'N/A')}, Expected features: {getattr(mdt_model, 'n_features_in_', 'N/A')}")

    # STEP 2: Read & Preprocess Frames
    print(f"\n[Step 2/5] Reading and preprocessing frames from directory...")
    print(f"  -> Target directory: {frames_directory}")
    t0 = time.time()
    
    raw_filenames = sorted(os.listdir(frames_directory))
    valid_extensions = ('.jpg', '.png', '.tif', '.tiff', '.bmp')
    image_files = [f for f in raw_filenames if f.lower().endswith(valid_extensions)]

    if not image_files:
        raise ValueError(f"No valid image files found in directory: {frames_directory}")

    frames = []
    for filename in image_files:
        file_path = os.path.join(frames_directory, filename)
        frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            preprocessed = preprocess_frame(frame)
            frames.append(preprocessed)
        else:
            print(f"  -> [Warning] Failed to read image: {filename}")

    frames = np.array(frames)
    print(f"  -> Successfully loaded and normalized {len(frames)} frames in {time.time()-t0:.2f}s")

    # STEP 3: Feature Extraction (Optical Flow)
    print("\n[Step 3/5] Extracting optical flow features across consecutive frames...")
    t0 = time.time()
    features = extract_features(frames)
    print(f"  -> Extracted {features.shape[0]} feature vectors of dimension {features.shape[1]} in {time.time()-t0:.2f}s")

    # STEP 4: Anomaly Scoring & Thresholding
    print("\n[Step 4/5] Computing GMM anomaly scores and classifying frames...")
    t0 = time.time()
    anomaly_scores = compute_anomaly_scores(mdt_model, features)
    
    # Calculate score statistics
    min_score, max_score = np.min(anomaly_scores), np.max(anomaly_scores)
    mean_score, std_score = np.mean(anomaly_scores), np.std(anomaly_scores)
    
    # Apply thresholding (adaptive mean + 0.3 * std or absolute threshold)
    processed_predictions, threshold_used = further_processing(anomaly_scores, threshold=None)
    
    print(f"  -> Anomaly score stats -> Min: {min_score:.1f}, Max: {max_score:.1f}, Mean: {mean_score:.1f}, Std: {std_score:.1f}")
    print(f"  -> Threshold applied: {threshold_used:.1f}")
    print(f"  -> Classification completed in {time.time()-t0:.2f}s")

    if args.start_frame is not None or args.end_frame is not None:
        if args.start_frame is None or args.end_frame is None:
            raise ValueError("Both --start-frame and --end-frame must be provided together.")
        if args.start_frame < 1 or args.end_frame < 1:
            raise ValueError("Frame numbers must be 1-based and greater than 0.")
        if args.start_frame > args.end_frame:
            raise ValueError("--start-frame must be less than or equal to --end-frame.")
        if args.end_frame > len(frames):
            raise ValueError(f"--end-frame exceeds available frames ({len(frames)} frames).")

        window_start = args.start_frame
        window_end = args.end_frame
        window_frames = frames[window_start - 1:window_end]

        if len(window_frames) < 2:
            raise ValueError("Selected frame window must contain at least two frames.")

        window_features = extract_features(window_frames)
        window_scores = compute_anomaly_scores(mdt_model, window_features)
        window_predictions, window_threshold = further_processing(window_scores, threshold=None)
        print_score_summary(window_scores, window_predictions, window_threshold,
                            start_frame=window_start, end_frame=window_end)

    # STEP 5: Results & Summary
    print("\n[Step 5/6] Prediction Classification Summary:")
    print("=" * 60)
    num_anomalies = np.sum(processed_predictions)
    total_eval = len(processed_predictions)
    anomaly_indices = np.where(processed_predictions == 1)[0] + 2  # +2 because frame 1-2 gives flow 1 (eval frame 2)
    
    print(f" Total evaluated frame transitions: {total_eval}")
    print(f" Normal frames predicted          : {total_eval - num_anomalies}")
    print(f" Anomaly frames predicted         : {num_anomalies}")
    
    if num_anomalies > 0:
        print(f" Anomaly detected in frame range  : Frame {anomaly_indices[0]} to Frame {anomaly_indices[-1]}")
    else:
        print(" No anomalies detected in this clip.")
        
    print("\nProcessed Predictions Array (0: Normal, 1: Anomaly):")
    print(processed_predictions)
    print("=" * 60)

    # STEP 6: Render Output Video & Visual Frame Overlay
    print("\n[Step 6/6] Rendering annotated video output and exporting keyframes...")
    t0 = time.time()
    
    output_video_path = os.path.join(base_dir, 'output_prediction.mp4')
    anomaly_only_video_path = os.path.join(base_dir, 'output_anomaly_only.mp4')
    sample_normal_path = os.path.join(base_dir, 'output_normal_sample.jpg')
    sample_anomaly_path = os.path.join(base_dir, 'output_anomaly_sample.jpg')

    # Read original frame dimensions
    first_orig = cv2.imread(os.path.join(frames_directory, image_files[0]))
    h, w = first_orig.shape[:2]
    # Upscale 2x for clear display
    render_w, render_h = w * 2, h * 2

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = args.fps  # adjustable playback speed (e.g. 30.0 or 60.0 for faster video)
    out_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (render_w, render_h))
    anomaly_only_writer = cv2.VideoWriter(anomaly_only_video_path, fourcc, output_fps, (render_w, render_h))

    show_gui = args.show
    show_anomaly_popup = args.popup
    saved_normal_sample = False
    saved_anomaly_sample = False
    anomaly_frames_written = 0
    anomaly_frame_dir = os.path.join(base_dir, 'anomaly_frames')
    os.makedirs(anomaly_frame_dir, exist_ok=True)

    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

    for idx, filename in enumerate(image_files):
        file_path = os.path.join(frames_directory, filename)
        orig_bgr = cv2.imread(file_path)
        if orig_bgr is None:
            continue
            
        render_frame = cv2.resize(orig_bgr, (render_w, render_h))
        
        # Frame 0 is initial frame, transitions start at idx-1
        if idx == 0:
            is_anomaly = False
            score_val = anomaly_scores[0] if len(anomaly_scores) > 0 else 0.0
        else:
            pred_idx = min(idx - 1, len(processed_predictions) - 1)
            is_anomaly = bool(processed_predictions[pred_idx] == 1)
            score_val = float(anomaly_scores[pred_idx])

        # Header banner (Top 40px)
        banner_color = (0, 0, 220) if is_anomaly else (0, 160, 0)
        status_text = "ANOMALY DETECTED" if is_anomaly else "NORMAL CROWD"
        
        # Overlay semi-transparent banner
        overlay = render_frame.copy()
        cv2.rectangle(overlay, (0, 0), (render_w, 42), banner_color, -1)
        cv2.addWeighted(overlay, 0.75, render_frame, 0.25, 0, render_frame)
        
        # Draw a tighter box around the moving anomaly region for anomaly frames
        if is_anomaly:
            box_color = (0, 0, 255)
            box_thickness = 3

            fg_mask = background_subtractor.apply(orig_bgr)
            fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                valid_boxes = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    x, y, w_box, h_box = cv2.boundingRect(contour)
                    aspect_ratio = w_box / float(h_box) if h_box > 0 else 0
                    is_person_like = 80 < area < 4000 and 0.3 < aspect_ratio < 2.0
                    if is_person_like:
                        x1 = max(0, int(x * 2 - 15))
                        y1 = max(0, int(y * 2 - 15))
                        x2 = min(render_w - 1, int(x * 2 + w_box * 2 + 15))
                        y2 = min(render_h - 1, int(y * 2 + h_box * 2 + 15))
                        valid_boxes.append((x1, y1, x2, y2))

                if valid_boxes:
                    for idx_box, (x1, y1, x2, y2) in enumerate(valid_boxes, start=1):
                        cv2.rectangle(render_frame, (x1, y1), (x2, y2), box_color, box_thickness)
                        cv2.rectangle(render_frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)
                        label = f"Anomaly {idx_box}"
                        cv2.putText(render_frame, label, (x1 + 5, max(15, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.rectangle(render_frame, (20, 20), (render_w - 21, render_h - 21), box_color, box_thickness)
            else:
                cv2.rectangle(render_frame, (20, 20), (render_w - 21, render_h - 21), box_color, box_thickness)

            cv2.putText(render_frame, "ANOMALY FRAME", (10, render_h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw status text & frame info
        cv2.putText(render_frame, f"STATUS: {status_text}", (10, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(render_frame, f"Frame {idx+1}/{len(image_files)}", (render_w - 140, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(render_frame, f"Score: {score_val:.2f}", (10, render_h - 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Save video frame
        out_writer.write(render_frame)
        if is_anomaly:
            anomaly_only_writer.write(render_frame)
            anomaly_frames_written += 1
            anomaly_frame_path = os.path.join(anomaly_frame_dir, f"anomaly_{idx+1:03d}.jpg")
            cv2.imwrite(anomaly_frame_path, render_frame)

        # Export sample images
        if not is_anomaly and not saved_normal_sample:
            cv2.imwrite(sample_normal_path, render_frame)
            saved_normal_sample = True
        elif is_anomaly and not saved_anomaly_sample:
            cv2.imwrite(sample_anomaly_path, render_frame)
            saved_anomaly_sample = True

        # Optional GUI Display
        if show_gui:
            cv2.namedWindow("Crowd Anomaly Detection - Live Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Crowd Anomaly Detection - Live Preview", render_w, render_h)
            cv2.imshow("Crowd Anomaly Detection - Live Preview", render_frame)
            if cv2.waitKey(max(1, int(1000 / output_fps))) & 0xFF in (ord('q'), 27):
                print("  -> Interrupted by user.")
                break

        if show_anomaly_popup and is_anomaly:
            popup_frame = render_frame.copy()
            cv2.rectangle(popup_frame, (0, 0), (render_w - 1, render_h - 1), (0, 0, 255), 8)
            cv2.putText(popup_frame, "ANOMALY FRAME", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(popup_frame, f"Frame {idx+1}/{len(image_files)}", (20, render_h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.namedWindow("Detected Anomaly Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detected Anomaly Frame", render_w, render_h)
            cv2.imshow("Detected Anomaly Frame", popup_frame)
            if cv2.waitKey(max(1, int(1000 / output_fps))) & 0xFF in (ord('q'), 27):
                print("  -> Interrupted by user.")
                break

    out_writer.release()
    if show_gui:
        cv2.destroyAllWindows()

    out_writer.release()
    anomaly_only_writer.release()
    if show_gui or show_anomaly_popup:
        cv2.destroyAllWindows()

    print(f"  -> Video rendering complete in {time.time()-t0:.2f}s")
    print(f"  -> Saved output video to: {output_video_path}")
    print(f"  -> Saved anomaly-only video to: {anomaly_only_video_path}")
    if saved_normal_sample:
        print(f"  -> Saved normal sample frame to : {sample_normal_path}")
    if saved_anomaly_sample:
        print(f"  -> Saved anomaly sample frame to: {sample_anomaly_path}")
    if anomaly_frames_written > 0:
        print(f"  -> Wrote {anomaly_frames_written} anomaly-highlighted frames to the anomaly-only video")
    else:
        print("  -> No anomaly frames were detected, so the anomaly-only video is empty")
    print("=" * 60)

if __name__ == '__main__':
    main()
