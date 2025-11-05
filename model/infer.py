"""
Inference:
- Slides a window to predict future frames of receiving hand.
- Generates video visualization of predicted future frames.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import torch
import cv2
from .config import CKPT_PATH, PRED_DIR, VIDEO_DIR, SEQ_LEN, FUTURE_FRAMES
from .data import load_features, load_receiving_hand_world
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

def create_video(predictions: list, frames: list, stem: str, fps: int = 30):
    """
    Create a video visualization of predicted future frames.
    Shows 3D hand landmarks projected to 2D.
    """
    if not predictions:
        print("No predictions to visualize")
        return
    
    # Get dimensions
    future_frames, out_dim = predictions[0].shape
    n_landmarks = out_dim // 3  # Assuming x,y,z for each landmark
    
    # Create video writer
    video_path = VIDEO_DIR / f"{stem}_predicted_future.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up video dimensions
    width, height = 800, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    # Create visualization for each prediction
    for pred_idx, (pred, frame) in enumerate(zip(predictions, frames)):
        # pred: [future_frames, out_dim] -> reshape to [future_frames, n_landmarks, 3]
        pred_3d = pred.reshape(future_frames, n_landmarks, 3)
        
        # Create a frame showing all future frames side by side or as animation
        # For simplicity, show each future frame as a separate frame in the video
        for f_idx in range(future_frames):
            frame_img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Get landmarks for this future frame
            landmarks = pred_3d[f_idx]  # [n_landmarks, 3]
            
            # Normalize and project to 2D screen coordinates
            # Center and scale the landmarks
            if landmarks.shape[0] > 0:
                center = landmarks.mean(axis=0)
                landmarks_centered = landmarks - center
                scale = np.abs(landmarks_centered).max() if np.abs(landmarks_centered).max() > 0 else 1.0
                if scale > 0:
                    landmarks_centered = landmarks_centered / scale * min(width, height) * 0.3
            
            # Project to 2D (use x,z as x,y for top-down view, or x,y for front view)
            screen_coords = []
            for lm in landmarks_centered:
                # Front view: x,y -> screen
                x = int(lm[0] + width // 2)
                y = int(-lm[1] + height // 2)  # Flip y for screen coordinates
                screen_coords.append((x, y))
            
            # Draw landmarks
            for i, (x, y) in enumerate(screen_coords):
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame_img, (x, y), 3, (0, 255, 0), -1)
            
            # Draw connections (simplified hand skeleton)
            # Wrist to finger bases
            if len(screen_coords) >= 5:
                wrist = screen_coords[0]
                for i in range(1, 5):
                    if i < len(screen_coords):
                        cv2.line(frame_img, wrist, screen_coords[i], (255, 255, 255), 1)
            
            # Add text overlay
            cv2.putText(frame_img, f"Frame {frame} -> Future {f_idx+1}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_img, f"Prediction {pred_idx+1}/{len(predictions)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writer.write(frame_img)
    
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
