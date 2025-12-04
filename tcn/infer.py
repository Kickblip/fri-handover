"""
Inference with TCN model (Standalone).
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import torch
import sys
import os
import cv2

# Fix path to find local config
sys.path.append(str(Path(__file__).resolve().parent))

# Try importing Kinect library
try:
    from pyk4a import PyK4APlayback, CalibrationType
except ImportError:
    PyK4APlayback = None

# --- IMPORTS ---
from tcn_config import (PRED_DIR, VIDEO_DIR, SEQ_LEN, FUTURE_FRAMES, CKPT_DIR, ROOT)
import data
from model import TemporalConvNet
# ---------------

def load_checkpoint(path: Path, device: str):
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt['config']
    
    model = TemporalConvNet(
        in_dim=cfg['in_dim'],
        num_channels=cfg['num_channels'],
        out_dim=cfg['out_dim'],
        future_frames=FUTURE_FRAMES,
        kernel_size=cfg.get('kernel_size', 3)
    ).to(device)
    
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

@torch.no_grad()
def predict_future_frames(stem: str, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use local data loader
    X_input, frames_input = data.load_features(stem)
    if X_input is None: raise ValueError(f"Could not load data for stem: {stem}")
    T = X_input.shape[0]
    
    # Point to the TCN checkpoint
    ckpt_path = CKPT_DIR / "tcn_handover.pt"
    model = load_checkpoint(ckpt_path, device)
    
    predictions = []
    pred_frames = []
    
    # BLOCK INFERENCE STRATEGY (Smoother visualization)
    for start in range(0, T - FUTURE_FRAMES, FUTURE_FRAMES):
        end = start + SEQ_LEN - 1
        if end >= T: break
        
        chunk = torch.from_numpy(X_input[start:end+1]).unsqueeze(0).float().to(device)
        pred = model(chunk)
        predictions.append(pred.cpu().numpy()[0])
        pred_frames.append(frames_input[end])
    
    return predictions, pred_frames

def get_camera_intrinsics(video_path: Path):
    if video_path.suffix.lower() == '.mkv' and PyK4APlayback:
        try:
            playback = PyK4APlayback(str(video_path))
            playback.open()
            K = playback.calibration.get_camera_matrix(CalibrationType.COLOR)
            playback.close()
            return K[0,0], K[1,1], K[0,2], K[1,2]
        except: pass
    
    cap = cv2.VideoCapture(str(video_path))
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    # Default Azure Kinect 1080p
    return 979.0, 979.0, w/2, h/2

def project_to_image(X, Y, Z, fx, fy, cx, cy):
    if Z <= 0: return None
    u = int((fx * X / Z) + cx)
    v = int((fy * Y / Z) + cy)
    return u, v

def create_video(predictions, frames, stem):
    # Find Original Video
    input_video_dir = ROOT / "input_video"
    video_path = input_video_dir / f"{stem}.mp4" 
    if not video_path.exists():
        video_path = input_video_dir / f"{stem}.mkv"
        
    if not video_path.exists():
        print(f"Error: Original video not found at {video_path}")
        return

    fx, fy, cx, cy = get_camera_intrinsics(video_path)
    
    # Load Ground Truth Data
    # NOTE: load_receiving_hand_world now correctly loads indices 0:63 (Hand 0)
    gt_coords, gt_frames = data.load_receiving_hand_world(stem)
    n_landmarks = 21
    gt_coords = gt_coords.reshape(len(gt_frames), n_landmarks, 3)
    gt_dict = {f: gt_coords[i] for i, f in enumerate(gt_frames)}
    
    # Setup Video Writer
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Ensure output directory exists
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"{stem}_tcn_pred.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Map predictions
    frame_map = {}
    for pred, start_frame in zip(predictions, frames):
        for i in range(len(pred)):
            target_frame = start_frame + 1 + i
            frame_map[target_frame] = pred[i].reshape(n_landmarks, 3)

    print(f"Rendering video to {out_path}...")
    curr_frame = 0
    
    while cap.isOpened():
        ret, img = cap.read()
        if not ret: break
        
        # Draw Actual Receiver (GREEN)
        if curr_frame in gt_dict:
            for pt in gt_dict[curr_frame]:
                uv = project_to_image(pt[0], pt[1], pt[2], fx, fy, cx, cy)
                if uv: cv2.circle(img, uv, 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

        # Draw Predicted Receiver (RED)
        if curr_frame in frame_map:
            for pt in frame_map[curr_frame]:
                uv = project_to_image(pt[0], pt[1], pt[2], fx, fy, cx, cy)
                if uv: cv2.circle(img, uv, 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        
        # Legend
        cv2.putText(img, "Green: Actual Receiver", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "Red:   Predicted Receiver", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        writer.write(img)
        curr_frame += 1
        
    cap.release()
    writer.release()
    print("Done.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="video stem, e.g., 1_w_b")
    ap.add_argument("--video", action="store_true", help="Generate video visualization")
    args = ap.parse_args()

    print(f"Predicting with TCN for {args.stem}...")
    predictions, frames = predict_future_frames(args.stem)
    
    # Save CSV
    outp = PRED_DIR / f"{args.stem}_tcn_predictions.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    
    if args.video:
        create_video(predictions, frames, args.stem)

if __name__ == "__main__":
    main()