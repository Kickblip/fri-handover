"""
Inference:
- Slides a window to predict future frames of receiving hand.
- Generates video visualization of predicted future frames.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import re
from typing import Any
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for video generation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import mediapipe as mp
from pyk4a import PyK4APlayback, CalibrationType
from .config import CKPT_PATH, PRED_DIR, VIDEO_DIR, SEQ_LEN, FUTURE_FRAMES, HANDS_DIR, WORLD_DIR, ROOT
from .data import load_features, load_receiving_hand_world, _read_csv, _pick_col
from .model import HandoverTransformer

# MediaPipe hand connections for skeleton drawing
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)

def load_checkpoint(path: Path, device: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}\n"
            f"Please train the model first by running:\n"
            f"  python3 -m model.train"
        )
    ckpt = torch.load(path, map_location=device)
    model = HandoverTransformer(
        in_dim=ckpt["in_dim"],
        out_dim=ckpt["out_dim"],
        d_model=ckpt["cfg"]["D_MODEL"],
        nhead=ckpt["cfg"]["N_HEAD"],
        nlayers=ckpt["cfg"]["N_LAYERS"],
        ffdim=ckpt["cfg"]["FFN_DIM"],
        dropout=ckpt["cfg"]["DROPOUT"],
        future_frames=ckpt["cfg"]["FUTURE_FRAMES"],
    ).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model

@torch.no_grad()
def predict_future_frames(stem: str, device=None):
    """
    Predict future frames using non-overlapping windows that advance by FUTURE_FRAMES:
    - Uses frames 0-9 to predict frames 10-29 
    - Uses frames 20-29 to predict frames 30-49
    - Uses frames 40-49 to predict frames 50-69
    - Each prediction is independent (uses only original input data, not previous predictions)
    - Windows advance by FUTURE_FRAMES (20) each time
    
    Returns:
        predictions: List of [future_frames, out_dim] arrays, one per valid position
        frames: List of frame indices where predictions were made (last frame of input window)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_input, frames_input = load_features(stem)  # [T, D_in]
    T = X_input.shape[0]
    L = SEQ_LEN
    
    model = load_checkpoint(CKPT_PATH, device)
    
    predictions = []
    pred_frames = []
    
    # Predict using non-overlapping windows that advance by FUTURE_FRAMES each time
    # Window 0: frames 0-9 → predict 10-29
    # Window 1: frames 20-29 → predict 30-49
    # Window 2: frames 40-49 → predict 50-69
    # etc.v
    for start in range(0, T - FUTURE_FRAMES, FUTURE_FRAMES):
        end = start + L - 1
        if end >= T:
            break
        chunk = torch.from_numpy(X_input[start:end+1]).unsqueeze(0).float().to(device)  # [1, L, D_in]
        pred = model(chunk)  # [1, future_frames, D_out]
        predictions.append(pred.cpu().numpy()[0])  # [future_frames, D_out]
        pred_frames.append(frames_input[end])
    
    return predictions, pred_frames

def load_giving_hand_world(stem: str) -> tuple[np.ndarray, list[int]]:
    """
    Load giving hand (hand_1) world coordinates.
    Returns world coordinates for all 21 landmarks (63 features: x,y,z for each).
    Uses the same consistency fix as load_both_hands_world to ensure hand1 stays consistent.
    
    Expected CSV format: frame_idx, h1_lm0_x, h1_lm0_y, h1_lm0_z, h1_lm1_x, ...
    """
    # Load both hands with consistency fix, then extract only hand1 (giving hand)
    # This ensures hand1 is consistently labeled throughout the video
    from .data import load_both_hands_world
    X_both, frames = load_both_hands_world(stem)
    
    # Extract hand1 (second half: indices 63-125) - this is the giving hand
    X1 = X_both[:, 63:126]  # [T, 63]
    
    return X1, frames

def find_original_video(stem: str) -> Path:
    """Find the original video file by stem name."""
    input_video_dir = ROOT / "input_video"
    # Try common video extensions
    for ext in [".mkv", ".mp4", ".avi", ".mov"]:
        video_path = input_video_dir / f"{stem}{ext}"
        if video_path.exists():
            return video_path
    
    # Fallback: try world video from mediapipe outputs
    world_video = ROOT / "mediapipe_outputs" / "video" / "world" / f"{stem}_world.mp4"
    if world_video.exists():
        return world_video
    
    raise FileNotFoundError(
        f"Could not find original video for stem '{stem}'. "
        f"Tried: {input_video_dir / f'{stem}.mkv'}, {input_video_dir / f'{stem}.mp4'}, "
        f"and {world_video}"
    )

def get_camera_intrinsics(video_path: Path) -> tuple[float, float, float, float]:
    """
    Get camera intrinsics from Kinect MKV file or use defaults.
    Returns: (fx, fy, cx, cy)
    """
    # Try to load from Kinect MKV file
    if video_path.suffix.lower() == '.mkv':
        try:
            playback = PyK4APlayback(str(video_path))
            playback.open()
            calib = playback.calibration
            K = calib.get_camera_matrix(CalibrationType.COLOR)
            playback.close()
            
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            print(f"Loaded camera intrinsics from Kinect MKV: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
            return fx, fy, cx, cy
        except Exception as e:
            print(f"Warning: Could not load calibration from MKV file: {e}")
            print("Using default Kinect Azure intrinsics for 1080p")
    
    # Default Kinect Azure intrinsics for 1080p (RES_1080P)
    # Typical values for Kinect Azure at 1080p:
    # fx ≈ 979.0, fy ≈ 979.0, cx ≈ 960.0, cy ≈ 540.0
    # But we'll use the video dimensions to compute cx, cy
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Kinect Azure 1080p typical intrinsics
        fx = fy = 979.0  # Approximate focal length for 1080p
        cx = width / 2.0
        cy = height / 2.0
        print(f"Using default Kinect Azure intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        return fx, fy, cx, cy
    
    # Fallback to old hardcoded values
    print("Warning: Using fallback intrinsics (may be inaccurate)")
    return 600.0, 600.0, 960.0, 540.0

def project_to_image(X: float, Y: float, Z: float, fx: float, fy: float, cx: float, cy: float) -> tuple[int, int] | None:
    """Project 3D world coordinates to 2D image coordinates using camera intrinsics."""
    if Z <= 0:
        return None
    
    x_pix = int((fx * X / Z) + cx)
    y_pix = int((fy * Y / Z) + cy)
    
    return x_pix, y_pix

def create_video(predictions: list, frames: list, stem: str, fps: int = 30):
    """
    Create a video visualization with:
    - Predicted receiving hand (hand_0) 21 points in RED
    - Actual receiving hand (hand_0) 21 points in GREEN from original video
    """
    if not predictions:
        print("No predictions to visualize")
        return
    
    # Get dimensions
    future_frames, out_dim = predictions[0].shape
    n_landmarks = out_dim // 3  # Assuming x,y,z for each landmark (should be 21)
    
    # Load original video
    original_video_path = find_original_video(stem)
    print(f"Loading original video: {original_video_path}")
    cap = cv2.VideoCapture(str(original_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {original_video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get camera intrinsics from video file
    print(f"Loading camera intrinsics...")
    fx, fy, cx, cy = get_camera_intrinsics(original_video_path)
    
    # Load actual receiving hand (hand_0) world coordinates
    print(f"Loading actual receiving hand (hand_0) coordinates...")
    from .data import load_receiving_hand_world
    receiving_hand_coords, receiving_frames = load_receiving_hand_world(stem)
    # Reshape to [T, 21, 3]
    receiving_hand_coords = receiving_hand_coords.reshape(len(receiving_frames), n_landmarks, 3)
    
    # Create a mapping from frame index to actual receiving hand coordinates
    receiving_hand_dict = {f: receiving_hand_coords[i] for i, f in enumerate(receiving_frames)}
    
    # Create video writer
    video_path = VIDEO_DIR / f"{stem}_predicted_future.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (width, height))
    
    # Create a mapping from frame index to predicted hand coordinates
    # Each prediction step predicts FUTURE_FRAMES frames starting from (frame + 1)
    # pred_frame is the last frame of the input window, prediction starts at pred_frame + 1
    frame_to_prediction = {}  # Maps frame_idx -> [n_landmarks, 3]
    for pred, pred_frame_end in zip(predictions, frames):
        # pred: [future_frames, out_dim] -> reshape to [future_frames, n_landmarks, 3]
        pred_3d = pred.reshape(future_frames, n_landmarks, 3)
        # This prediction covers frames from (pred_frame_end + 1) to (pred_frame_end + FUTURE_FRAMES)
        for i in range(future_frames):
            frame_idx = pred_frame_end + 1 + i
            frame_to_prediction[frame_idx] = pred_3d[i]  # [n_landmarks, 3]
    
    current_frame_idx = 0
    
    try:
        while True:
            ret, frame_img = cap.read()
            if not ret:
                break
            
            current_frame_idx += 1
            
            # Draw actual receiving hand (hand_1) in GREEN from original video
            if current_frame_idx in receiving_hand_dict:
                actual_hand = receiving_hand_dict[current_frame_idx]  # [21, 3]
                for landmark in actual_hand:
                    X, Y, Z = landmark[0], landmark[1], landmark[2]
                    if np.isnan(X) or np.isnan(Y) or np.isnan(Z):
                        continue
                    uv = project_to_image(X, Y, Z, fx, fy, cx, cy)
                    if uv is not None:
                        u, v = uv
                        if 0 <= u < width and 0 <= v < height:
                            cv2.circle(frame_img, (u, v), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            
            # Draw predicted receiving hand (hand_0) in RED
            # Look up prediction for this frame
            if current_frame_idx in frame_to_prediction:
                predicted_hand = frame_to_prediction[current_frame_idx]  # [21, 3]
                for landmark in predicted_hand:
                    X, Y, Z = landmark[0], landmark[1], landmark[2]
                    if np.isnan(X) or np.isnan(Y) or np.isnan(Z):
                        continue
                    uv = project_to_image(X, Y, Z, fx, fy, cx, cy)
                    if uv is not None:
                        u, v = uv
                        if 0 <= u < width and 0 <= v < height:
                            cv2.circle(frame_img, (u, v), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            
            writer.write(frame_img)
    
    finally:
        cap.release()
        writer.release()
    
    print(f"✓ Created video: {video_path.resolve()}")

def create_3d_video(predictions: list, frames: list, stem: str, fps: int = 30):
    """
    Create a 3D matplotlib visualization video with:
    - Actual receiving hand (hand_0) in GREEN
    - Predicted receiving hand (hand_0) in RED
    - Shows 3D skeleton using MediaPipe connections
    """
    if not predictions:
        print("No predictions to visualize")
        return
    
    # Get dimensions
    future_frames, out_dim = predictions[0].shape
    n_landmarks = out_dim // 3  # Should be 21
    
    # Load actual receiving hand coordinates
    print(f"Loading actual receiving hand (hand_0) coordinates...")
    from .data import load_receiving_hand_world
    receiving_hand_coords, receiving_frames = load_receiving_hand_world(stem)
    # Reshape to [T, 21, 3]
    receiving_hand_coords = receiving_hand_coords.reshape(len(receiving_frames), n_landmarks, 3)
    
    # Create mapping from frame index to actual hand coordinates
    receiving_hand_dict = {f: receiving_hand_coords[i] for i, f in enumerate(receiving_frames)}
    
    # Create mapping from frame index to predicted hand coordinates
    frame_to_prediction = {}  # Maps frame_idx -> [n_landmarks, 3]
    for pred, pred_frame_end in zip(predictions, frames):
        pred_3d = pred.reshape(future_frames, n_landmarks, 3)
        for i in range(future_frames):
            frame_idx = pred_frame_end + 1 + i
            frame_to_prediction[frame_idx] = pred_3d[i]  # [n_landmarks, 3]
    
    # Get all frames that have either actual or predicted data
    all_frames = set(receiving_hand_dict.keys()) | set(frame_to_prediction.keys())
    if not all_frames:
        print("No frames with actual or predicted data")
        return
    
    # Compute axis limits from all data
    all_coords = []
    for coords_dict in [receiving_hand_dict, frame_to_prediction]:
        for frame_idx, coords in coords_dict.items():
            valid = ~np.isnan(coords).any(axis=1)
            if valid.any():
                all_coords.append(coords[valid])
    
    if not all_coords:
        print("No valid coordinates found")
        return
    
    all_coords_array = np.concatenate(all_coords, axis=0)
    xmin, xmax = all_coords_array[:, 0].min(), all_coords_array[:, 0].max()
    ymin, ymax = all_coords_array[:, 1].min(), all_coords_array[:, 1].max()
    zmin, zmax = all_coords_array[:, 2].min(), all_coords_array[:, 2].max()
    
    # Add margin
    margin = 0.1
    xmin -= margin
    xmax += margin
    ymin -= margin
    ymax += margin
    zmin -= margin
    zmax += margin
    
    # Create video writer
    video_path = VIDEO_DIR / f"{stem}_predicted_future_3d.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Video dimensions
    W, H = 800, 800
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))
    
    # Create figure
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasAgg(fig)
    
    # Process each frame
    sorted_frames = sorted(all_frames)
    print(f"Generating 3D video for {len(sorted_frames)} frames...")
    
    for frame_idx in sorted_frames:
        ax.clear()
        
        # Set title and labels
        ax.set_title(f"3D Hand Prediction - Frame {frame_idx}", fontsize=14)
        ax.set_xlabel("X (m)", fontsize=12)
        ax.set_ylabel("Y (m)", fontsize=12)
        ax.set_zlabel("Z (m)", fontsize=12)
        
        # Set axis limits
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        # Track if we've added labels to legend
        has_actual = False
        has_predicted = False
        
        # Draw actual hand in GREEN
        if frame_idx in receiving_hand_dict:
            actual_hand = receiving_hand_dict[frame_idx]  # [21, 3]
            valid_actual = ~np.isnan(actual_hand).any(axis=1)
            
            if valid_actual.any():
                actual_valid = actual_hand[valid_actual]
                label = 'Actual' if not has_actual else ''
                ax.scatter(actual_valid[:, 0], actual_valid[:, 1], actual_valid[:, 2], 
                          c='green', s=50, alpha=0.8, label=label)
                has_actual = True
                
                # Draw skeleton connections
                for connection in HAND_CONNECTIONS:
                    i1, i2 = connection[0], connection[1]
                    if (i1 < n_landmarks and i2 < n_landmarks and 
                        valid_actual[i1] and valid_actual[i2]):
                        ax.plot([actual_hand[i1, 0], actual_hand[i2, 0]],
                               [actual_hand[i1, 1], actual_hand[i2, 1]],
                               [actual_hand[i1, 2], actual_hand[i2, 2]],
                               'g-', linewidth=2, alpha=0.6)
        
        # Draw predicted hand in RED
        if frame_idx in frame_to_prediction:
            predicted_hand = frame_to_prediction[frame_idx]  # [21, 3]
            valid_pred = ~np.isnan(predicted_hand).any(axis=1)
            
            if valid_pred.any():
                pred_valid = predicted_hand[valid_pred]
                label = 'Predicted' if not has_predicted else ''
                ax.scatter(pred_valid[:, 0], pred_valid[:, 1], pred_valid[:, 2], 
                          c='red', s=50, alpha=0.8, label=label, marker='^')
                has_predicted = True
                
                # Draw skeleton connections
                for connection in HAND_CONNECTIONS:
                    i1, i2 = connection[0], connection[1]
                    if (i1 < n_landmarks and i2 < n_landmarks and 
                        valid_pred[i1] and valid_pred[i2]):
                        ax.plot([predicted_hand[i1, 0], predicted_hand[i2, 0]],
                               [predicted_hand[i1, 1], predicted_hand[i2, 1]],
                               [predicted_hand[i1, 2], predicted_hand[i2, 2]],
                               'r--', linewidth=2, alpha=0.6)
        
        # Add legend if we have any data
        if has_actual or has_predicted:
            ax.legend(loc='upper right')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Render to image
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        img_rgba = buf.reshape((H, W, 4))
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
        
        writer.write(img_bgr)
    
    writer.release()
    plt.close(fig)
    print(f"✓ Created 3D video: {video_path.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="video stem, e.g., 1_w_b")
    ap.add_argument("--video", action="store_true", help="Generate 2D video visualization (overlay on original video)")
    ap.add_argument("--video3d", action="store_true", help="Generate 3D matplotlib video visualization")
    args = ap.parse_args()

    print(f"Predicting future frames for {args.stem}...")
    predictions, frames = predict_future_frames(args.stem)
    
    # Save predictions as CSV
    outp = PRED_DIR / f"{args.stem}_future_predictions.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    
    with outp.open("w") as f:
        f.write("frame,future_frame_idx")
        # Write header for all coordinates
        future_frames, out_dim = predictions[0].shape
        n_landmarks = out_dim // 3
        for lm_idx in range(n_landmarks):
            f.write(f",lm_{lm_idx}_x,lm_{lm_idx}_y,lm_{lm_idx}_z")
        f.write("\n")
        
        for pred, frame in zip(predictions, frames):
            for f_idx in range(future_frames):
                f.write(f"{int(frame)},{f_idx}")
                for lm_idx in range(n_landmarks):
                    x, y, z = pred[f_idx, lm_idx*3:(lm_idx+1)*3]
                    f.write(f",{x:.6f},{y:.6f},{z:.6f}")
                f.write("\n")
    
    print(f"✓ Wrote predictions: {outp.resolve()}")
    
    if args.video:
        print("Generating 2D video visualization...")
        create_video(predictions, frames, args.stem)
    
    if args.video3d:
        print("Generating 3D video visualization...")
        create_3d_video(predictions, frames, args.stem)

if __name__ == "__main__":
    main()
    
