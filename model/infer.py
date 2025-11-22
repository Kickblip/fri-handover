"""
Inference:
- Slides a window to predict future frames of receiving hand.
- Generates video visualization of predicted future frames.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import torch
import cv2
from .config import CKPT_PATH, PRED_DIR, VIDEO_DIR, SEQ_LEN, FUTURE_FRAMES, HANDS_DIR, WORLD_DIR, ROOT
from .data import load_features, load_receiving_hand_world, _read_csv, _pick_col
from .model import HandoverTransformer

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
    Predict future frames for all valid positions in the sequence.
    
    Returns:
        predictions: List of [future_frames, out_dim] arrays, one per valid position
        frames: List of frame indices where predictions were made
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_input, frames_input = load_features(stem)  # [T, D_in]
    T = X_input.shape[0]
    L = SEQ_LEN
    
    model = load_checkpoint(CKPT_PATH, device)
    
    predictions = []
    pred_frames = []
    
    # Predict for all valid positions
    for end in range(L-1, T - FUTURE_FRAMES):
        start = end - (L-1)
        chunk = torch.from_numpy(X_input[start:end+1]).unsqueeze(0).float().to(device)  # [1, L, D_in]
        pred = model(chunk)  # [1, future_frames, D_out]
        predictions.append(pred.cpu().numpy()[0])  # [future_frames, D_out]
        pred_frames.append(frames_input[end])
    
    return predictions, pred_frames

def load_giving_hand_world(stem: str) -> tuple[np.ndarray, list[int]]:
    """
    Load giving hand (hand_0) world coordinates.
    Returns world coordinates for all 21 landmarks (63 features: x,y,z for each).
    
    Tries new format first: {stem}_hands.csv from HANDS_DIR
    Falls back to old format: {stem}_world.csv from WORLD_DIR
    """
    # Try new format first
    p = HANDS_DIR / f"{stem}_hands.csv" if HANDS_DIR.exists() else None
    if p is None or not p.exists():
        # Fallback to old format
        p = WORLD_DIR / f"{stem}_world.csv"
    
    if not p.exists():
        raise FileNotFoundError(f"Missing hands CSV: {p}")
    
    df = _read_csv(p)
    fcol = _pick_col(df, "frame", ["frame_index", "frame_idx"])
    frames = df[fcol].astype(int).tolist()
    
    # Extract all hand_0 world coordinate columns (ending with _0)
    hand0_cols = [c for c in df.columns if c.endswith("_world_x_0") or 
                  c.endswith("_world_y_0") or c.endswith("_world_z_0")]
    
    if not hand0_cols:
        # Fallback: try to find any columns with _0 suffix and x/y/z pattern
        hand0_cols = [c for c in df.columns if re.search(r"_world_[xyz]_0$", c)]
    
    if not hand0_cols:
        all_cols = list(df.columns)
        print(f"Available columns in {p.name}: {all_cols[:10]}... (showing first 10)")
        raise ValueError(f"No hand_0 world coordinates found in {p}. Expected columns ending with '_world_x_0', '_world_y_0', or '_world_z_0'")
    
    # Sort columns to ensure consistent ordering (x, y, z for each landmark)
    hand0_cols = sorted(hand0_cols)
    
    # Extract features - convert to numeric, coercing errors to NaN
    X = df[hand0_cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    
    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)
    
    return X, frames

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

def project_to_image(X: float, Y: float, Z: float, width: int, height: int) -> tuple[int, int] | None:
    """Project 3D world coordinates to 2D image coordinates using camera intrinsics."""
    # Camera intrinsics - adjust if your setup differs
    fx, fy = 600, 600
    cx, cy = width // 2, height // 2  # Center for video
    
    if Z <= 0:
        return None
    
    x_pix = int((fx * X / Z) + cx)
    y_pix = int((fy * Y / Z) + cy)
    
    return x_pix, y_pix

def create_video(predictions: list, frames: list, stem: str, fps: int = 30):
    """
    Create a video visualization with:
    - Predicted receiving hand (hand_1) 21 points in RED
    - Giving hand (hand_0) 21 points in GREEN on the original video
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
    
    # Load giving hand (hand_0) world coordinates
    print(f"Loading giving hand (hand_0) coordinates...")
    giving_hand_coords, giving_frames = load_giving_hand_world(stem)
    # Reshape to [T, 21, 3]
    giving_hand_coords = giving_hand_coords.reshape(len(giving_frames), n_landmarks, 3)
    
    # Create a mapping from frame index to giving hand coordinates
    giving_hand_dict = {f: giving_hand_coords[i] for i, f in enumerate(giving_frames)}
    
    # Create video writer
    video_path = VIDEO_DIR / f"{stem}_predicted_future.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (width, height))
    
    # Create a mapping from prediction frame to predictions
    # Each prediction is for a specific frame and predicts future_frames ahead
    pred_dict = {}
    for pred, frame in zip(predictions, frames):
        # pred: [future_frames, out_dim] -> reshape to [future_frames, n_landmarks, 3]
        pred_3d = pred.reshape(future_frames, n_landmarks, 3)
        pred_dict[frame] = pred_3d
    
    current_frame_idx = 0
    
    try:
        while True:
            ret, frame_img = cap.read()
            if not ret:
                break
            
            current_frame_idx += 1
            
            # Draw giving hand (hand_0) in GREEN from original video
            if current_frame_idx in giving_hand_dict:
                giving_hand = giving_hand_dict[current_frame_idx]  # [21, 3]
                for landmark in giving_hand:
                    X, Y, Z = landmark[0], landmark[1], landmark[2]
                    if np.isnan(X) or np.isnan(Y) or np.isnan(Z):
                        continue
                    uv = project_to_image(X, Y, Z, width, height)
                    if uv is not None:
                        u, v = uv
                        if 0 <= u < width and 0 <= v < height:
                            cv2.circle(frame_img, (u, v), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            
            # Draw predicted receiving hand (hand_1) in RED
            # Use the most recent prediction that covers this frame
            best_pred_frame = None
            best_future_idx = None
            
            for pred_frame, pred_3d in pred_dict.items():
                # Check if current_frame_idx is within the future frames of this prediction
                if pred_frame < current_frame_idx <= pred_frame + future_frames:
                    future_idx = current_frame_idx - pred_frame - 1  # Which future frame this is
                    if 0 <= future_idx < future_frames:
                        # Use the most recent prediction (largest pred_frame that still covers this frame)
                        if best_pred_frame is None or pred_frame > best_pred_frame:
                            best_pred_frame = pred_frame
                            best_future_idx = future_idx
            
            # Draw the best prediction if found
            if best_pred_frame is not None:
                predicted_hand = pred_dict[best_pred_frame][best_future_idx]  # [21, 3]
                for landmark in predicted_hand:
                    X, Y, Z = landmark[0], landmark[1], landmark[2]
                    if np.isnan(X) or np.isnan(Y) or np.isnan(Z):
                        continue
                    uv = project_to_image(X, Y, Z, width, height)
                    if uv is not None:
                        u, v = uv
                        if 0 <= u < width and 0 <= v < height:
                            cv2.circle(frame_img, (u, v), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            
            writer.write(frame_img)
    
    finally:
        cap.release()
        writer.release()
    
    print(f"✓ Created video: {video_path.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="video stem, e.g., 1_w_b")
    ap.add_argument("--video", action="store_true", help="Generate video visualization")
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
        print("Generating video visualization...")
        create_video(predictions, frames, args.stem)

if __name__ == "__main__":
    main()
