"""
Data loading utilities for the Handover Transformer.

This module:
- Loads Rodrigues rotation vectors per hand (orientation features)
- Creates proxy labels from the global minimum hand distance
- Generates overlapping frame sequences for training the transformer
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

from .config import *

# ------------------------- Helpers -------------------------

def _read_csv(p: Path):
    """Read a CSV file safely and raise a clear error if missing."""
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def list_stems():
    """List all video stems based on available Rodrigues CSV files."""
    return [p.stem.replace("_rodrigues", "") for p in RODR_DIR.glob("*_rodrigues.csv")]

# ---------------------- Feature loader ----------------------

def load_rodrigues(stem: str):
    """
    Load Rodrigues vectors for both hands for a given video.
    Returns:
        X: np.ndarray of shape [T, 2*D]  (concatenated left/right hand features)
        frames: list[int] of frame indices
    """
    df = _read_csv(RODR_DIR / f"{stem}_rodrigues.csv")

    # Expect columns like: frame, hand, rx, ry, rz
    if "frame" not in df.columns or "hand" not in df.columns:
        raise ValueError("Rodrigues CSV must have 'frame' and 'hand' columns.")

    # Identify all numeric Rodrigues columns
    feat_cols = [c for c in df.columns if c not in ("frame", "hand")]
    frames = sorted(df["frame"].unique())

    # For each frame, stack [hand0, hand1] features
    X = []
    for f in frames:
        sub = df[df["frame"] == f]
        h0 = sub[sub["hand"] == 0][feat_cols].to_numpy(dtype=np.float32)
        h1 = sub[sub["hand"] == 1][feat_cols].to_numpy(dtype=np.float32)
        # Pad with zeros if a hand is missing that frame
        h0v = h0[0] if len(h0) else np.zeros(len(feat_cols), np.float32)
        h1v = h1[0] if len(h1) else np.zeros(len(feat_cols), np.float32)
        X.append(np.concatenate([h0v, h1v]))

    X = np.stack(X)
    return X, frames

# --------------------- Proxy label maker ---------------------

def load_proxy_labels(stem: str, num_frames: int):
    """
    Create binary labels from the global minimum frame of closest_pair_distance.

    Positive (handover) = frames within ±5 of the global minimum.
    """
    p = GLOBAL_MIN_DIR / f"{stem}_closest_global.csv"
    if not p.exists():
        return np.zeros(num_frames, dtype=np.int64)
    df = pd.read_csv(p)

    if "frame" not in df.columns or len(df) == 0:
        return np.zeros(num_frames, dtype=np.int64)

    # Assume first row's 'frame' is the minimum-distance frame
    f = int(df.iloc[0]["frame"])
    y = np.zeros(num_frames, dtype=np.int64)
    lo, hi = max(0, f - 5), min(num_frames, f + 6)
    y[lo:hi] = 1
    return y

# ---------------------- Sequence builder ---------------------

def make_sequences(X, y, seq_len, stride):
    """
    Convert continuous frames into overlapping sequences of length seq_len.
    Each sequence is labeled by its final frame's label.
    """
    xs, ys = [], []
    for end in range(seq_len - 1, len(X), stride):
        start = end - (seq_len - 1)
        xs.append(X[start:end+1])
        ys.append(y[end])
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

# ---------------------- Dataset + loaders ---------------------

class HandoverDataset(Dataset):
    """Dataset that provides (sequence, label) pairs for training."""
    def __init__(self, stems, seq_len, stride):
        self.samples = []
        for s in stems:
            X, frames = load_rodrigues(s)
            y = load_proxy_labels(s, len(frames))
            xs, ys = make_sequences(X, y, seq_len, stride)
            for i in range(len(xs)):
                self.samples.append((xs[i], ys[i]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def split_stems():
    """Split all available stems into train/val/test subsets."""
    all_stems = list_stems()
    random.seed(1337)
    random.shuffle(all_stems)
    n = len(all_stems)
    n_train, n_val = int(0.7*n), int(0.15*n)
    return all_stems[:n_train], all_stems[n_train:n_train+n_val], all_stems[n_train+n_val:]

def build_loaders():
    """Build PyTorch DataLoaders for training/validation/test sets."""
    train, val, test = split_stems()
    make = lambda s: DataLoader(HandoverDataset(s, SEQ_LEN, SEQ_STRIDE),
                                batch_size=BATCH_SIZE, shuffle=True)
    return make(train), make(val), make(test)
