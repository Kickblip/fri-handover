"""
Inference utility:
- Slides a window with stride 1 to compute a probability per frame.
- (Optionally) saves a CSV "frame,prob" that you can visualize or post-process.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import torch

from .config import SEQ_LEN
from .data import load_features
from .model import HandoverTransformer

def load_checkpoint(path: Path, device: str):
    """
    Load a trained checkpoint and rebuild the model with its saved config.
    """
    ckpt = torch.load(path, map_location=device)
    model = HandoverTransformer(
        in_dim=ckpt["in_dim"],
        d_model=ckpt["cfg"]["D_MODEL"],
        nhead=ckpt["cfg"]["N_HEAD"],
        nlayers=ckpt["cfg"]["N_LAYERS"],
        ffdim=ckpt["cfg"]["FFN_DIM"],
        dropout=ckpt["cfg"]["DROPOUT"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    seq_len = ckpt["cfg"].get("SEQ_LEN", SEQ_LEN)
    return model, seq_len

@torch.no_grad()
def predict_stem(stem: str, ckpt_path="handover_best.pt", device=None):
    """
    Compute per-frame probabilities p_t by sliding a length-SEQ_LEN window across the video.
    We classify using the final frame of the window (causal, no future leakage).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load features (hands + vertices)
    X, frames = load_features(stem)        # X: [T, D], frames: [T]
    model, seq_len = load_checkpoint(Path(ckpt_path), device=device)

    T = X.shape[0]
    logits = np.full((T,), np.nan, dtype=np.float32)

    # Slide a window with stride=1 for smooth scores
    for end in range(seq_len - 1, T):
        start = end - (seq_len - 1)
        chunk = torch.from_numpy(X[start:end+1]).unsqueeze(0).float().to(device)  # [1, L, D]
        logits[end] = model(chunk).item()

    # For the first (seq_len-1) frames (no full context), copy the first valid score back
    logits[:seq_len - 1] = logits[seq_len - 1]

    # Convert logits to probabilities via sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))
    return frames, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="video stem, e.g., 1_video")
    ap.add_argument("--ckpt", default="handover_best.pt", help="checkpoint path")
    ap.add_argument("--out", default=None, help="optional CSV path to save 'frame,prob'")
    args = ap.parse_args()

    frames, probs = predict_stem(args.stem, ckpt_path=args.ckpt)
    print(f"[{args.stem}] frames={len(frames)} | mean prob={probs.mean():.3f} | first10={np.round(probs[:10],3)}")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w") as f:
            f.write("frame,prob\n")
            for fr, pr in zip(frames, probs):
                f.write(f"{fr},{pr:.6f}\n")
        print(f"Saved per-frame probabilities to: {outp.resolve()}")

if __name__ == "__main__":
    main()
