"""
Data utilities:
- Load and fuse features per frame:
    hands (Rodrigues) + object (vertices)
- Create proxy labels from closest_global (±5)
- Convert time-series into fixed-length windows for the transformer
- Expose PyTorch Datasets and DataLoaders with a stem-wise split (no leakage)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import RODR_DIR, VERTICES_DIR, GLOBAL_MIN_DIR, SEQ_LEN, SEQ_STRIDE, BATCH_SIZE

# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _read_csv(p: Path) -> pd.DataFrame:
    """Read CSV or fail early with a clear message (avoids silent shape bugs)."""
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")
    return pd.read_csv(p)

def list_stems() -> List[str]:
    """
    Discover stems by scanning Rodrigues files.
    Example: '1_video_rodrigues.csv' -> stem '1_video'.
    """
    return [p.stem.replace("_rodrigues", "") for p in RODR_DIR.glob("*_rodrigues.csv")]

# -----------------------------------------------------------------------------
# Feature loaders
# -----------------------------------------------------------------------------

def load_rodrigues(stem: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load per-frame Rodrigues vectors for up to 2 hands and stack them as [hand0 || hand1].
    Expected schema:
        columns: frame, hand, <rvecs...>
        where 'hand' ∈ {0,1}. Missing hand at a frame is zero-padded.

    Returns:
        X_hands: [T, 2*D_hand]  -- concatenated features per frame
        frames:  [T] list of frame indices
    """
    df = _read_csv(RODR_DIR / f"{stem}_rodrigues.csv")

    # Sanity checks
    if "frame" not in df.columns or "hand" not in df.columns:
        raise ValueError("Rodrigues CSV must contain 'frame' and 'hand' columns.")

    # All non-(frame,hand) columns are treated as numeric features
    feat_cols = [c for c in df.columns if c not in ("frame", "hand")]
    # Work strictly in sorted frame order
    frames = sorted(df["frame"].unique())

    X = []
    for f in frames:
        sub = df[df["frame"] == f]
        # Extract per-hand rows; padding with zeros if missing
        h0 = sub[sub["hand"] == 0][feat_cols].to_numpy(dtype=np.float32)
        h1 = sub[sub["hand"] == 1][feat_cols].to_numpy(dtype=np.float32)
        h0v = h0[0] if len(h0) else np.zeros(len(feat_cols), np.float32)
        h1v = h1[0] if len(h1) else np.zeros(len(feat_cols), np.float32)
        X.append(np.concatenate([h0v, h1v], axis=0))

    return np.stack(X, axis=0), frames  # [T, 2*D_hand], [T]

def load_vertices(stem: str, frames: List[int]) -> np.ndarray:
    """
    Load per-frame object vertices and align them to the provided frame list.
    - The file may have any number of vertex columns (flattened xyz triplets).
    - Missing frames are zero-filled (keeps shapes aligned).
    - If the file doesn't exist, returns an array with zero columns (feature-less).
    """
    p = VERTICES_DIR / f"{stem}_vertices.csv"
    if not p.exists():
        # Return zero-width block so concatenation still works
        return np.zeros((len(frames), 0), dtype=np.float32)

    df = pd.read_csv(p)
    if "frame" not in df.columns:
        raise ValueError(f"{p} must have a 'frame' column.")

    feat_cols = [c for c in df.columns if c != "frame"]
    if not feat_cols:
        return np.zeros((len(frames), 0), dtype=np.float32)

    # Index each row by frame for O(1) lookups while assembling
    rows = {int(r["frame"]): r[feat_cols].to_numpy(dtype=np.float32) for _, r in df.iterrows()}

    D = len(feat_cols)
    X = np.zeros((len(frames), D), dtype=np.float32)
    for i, f in enumerate(frames):
        if f in rows:
            X[i] = rows[f]  # if a frame is missing, it stays zeros
    return X  # [T, D_obj]

def load_features(stem: str) -> Tuple[np.ndarray, List[int]]:
    """
    Merge hand + object features along the last dimension.

    Returns:
        X: [T, 2*D_hand + D_obj]
        frames: [T] list of indices
    """
    X_hands, frames = load_rodrigues(stem)
    X_obj = load_vertices(stem, frames)
    X = np.concatenate([X_hands, X_obj], axis=1)  # zero-width concat ok if vertices missing
    return X, frames

# -----------------------------------------------------------------------------
# Labels (proxy supervision)
# -----------------------------------------------------------------------------

def load_proxy_labels(stem: str, num_frames: int) -> np.ndarray:
    """
    Build weak labels using the global-min distance frame:
      - mark frames in [min-5, min+5] as 1 (handover), others 0.

    This substitutes manual annotation and works surprisingly well for transfers.
    """
    p = GLOBAL_MIN_DIR / f"{stem}_closest_global.csv"
    if not p.exists():
        return np.zeros(num_frames, dtype=np.int64)

    df = pd.read_csv(p)
    if "frame" not in df.columns or len(df) == 0:
        return np.zeros(num_frames, dtype=np.int64)

    t_star = int(df.iloc[0]["frame"])
    y = np.zeros(num_frames, dtype=np.int64)
    lo, hi = max(0, t_star - 5), min(num_frames, t_star + 6)  # inclusive ±5
    y[lo:hi] = 1
    return y

# -----------------------------------------------------------------------------
# Windowing / Dataset / Dataloaders
# -----------------------------------------------------------------------------

def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, stride: int):
    """
    Convert frame-wise features into overlapping windows of length 'seq_len'.
    Each window is labeled by the label of its **last** frame (causal decision).
    """
    xs, ys = [], []
    for end in range(seq_len - 1, len(X), stride):
        start = end - (seq_len - 1)
        xs.append(X[start:end+1])
        ys.append(y[end])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

class HandoverDataset(Dataset):
    """
    PyTorch Dataset that yields (sequence, label) pairs.
    """
    def __init__(self, stems: List[str], seq_len: int, stride: int):
        self.samples = []
        for s in stems:
            X, frames = load_features(s)                       # [T,D]
            y = load_proxy_labels(s, len(frames))              # [T]
            xs, ys = make_sequences(X, y, seq_len, stride)     # [N,L,D], [N]
            # Flatten into per-window items for easy batching
            for i in range(len(xs)):
                self.samples.append((xs[i], ys[i]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def split_stems():
    """
    Create a deterministic 70/15/15 split by stem (video-level splitting prevents leakage).
    """
    stems = list_stems()
    random.seed(1337)
    random.shuffle(stems)
    n = len(stems)
    n_tr = int(0.7 * n)
    n_val = int(0.15 * n)
    return stems[:n_tr], stems[n_tr:n_tr+n_val], stems[n_tr+n_val:]

def build_loaders():
    """
    Return DataLoaders for train/val/test splits.
    """
    tr, va, te = split_stems()
    mk = lambda ss, shuf: DataLoader(HandoverDataset(ss, SEQ_LEN, SEQ_STRIDE),
                                     batch_size=BATCH_SIZE, shuffle=shuf, drop_last=False)
    return mk(tr, True), mk(va, False), mk(te, False)
