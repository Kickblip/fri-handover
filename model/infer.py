"""
Inference:
- Slides a window (stride=1) to compute a probability per frame.
- Writes dataset/model_output/predictions/<stem>_probs.csv (frame,prob).
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import torch
from .config import CKPT_PATH, PRED_DIR, SEQ_LEN
from .data import load_features
from .model import HandoverTransformer

def load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location=device)
    model = HandoverTransformer(
        in_dim=ckpt["in_dim"],
        d_model=ckpt["cfg"]["D_MODEL"],
        nhead=ckpt["cfg"]["N_HEAD"],
        nlayers=ckpt["cfg"]["N_LAYERS"],
        ffdim=ckpt["cfg"]["FFN_DIM"],
        dropout=ckpt["cfg"]["DROPOUT"],
    ).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    return model

@torch.no_grad()
def predict_stem(stem: str, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, frames = load_features(stem)        # [T,D]
    T = X.shape[0]; L = SEQ_LEN
    model = load_checkpoint(CKPT_PATH, device)
    logits = np.full((T,), np.nan, np.float32)
    for end in range(L-1, T):
        start = end - (L-1)
        chunk = torch.from_numpy(X[start:end+1]).unsqueeze(0).float().to(device)
        logits[end] = model(chunk).item()
    logits[:L-1] = logits[L-1]            # fill warm-up
    probs = 1.0 / (1.0 + np.exp(-logits))
    return frames, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem", help="video stem, e.g., 1_video")
    args = ap.parse_args()

    frames, probs = predict_stem(args.stem)
    outp = PRED_DIR / f"{args.stem}_probs.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        f.write("frame,prob\n")
        for fr, pr in zip(frames, probs):
            f.write(f"{int(fr)},{pr:.6f}\n")
    print(f"✓ wrote {outp.resolve()}")

if __name__ == "__main__":
    main()
