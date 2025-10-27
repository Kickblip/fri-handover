"""
Data utilities:
- Load per-frame features: Rodrigues (two hands concatenated) + Vertices.
- Build labels from closest_global: frames in [min-5, min+5] -> 1 else 0.
- Convert time-series into fixed-length windows.
- Stem-wise splits to prevent leakage.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .config import (RODR_DIR, VERTICES_DIR, GLOBAL_MIN_DIR,
                     SEQ_LEN, SEQ_STRIDE, BATCH_SIZE)

# ---------- helpers ----------
def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(f"Missing CSV: {p}")
    return pd.read_csv(p)

def _pick_col(df: pd.DataFrame, main: str, aliases: List[str]) -> str:
    for c in [main] + aliases:
        if c in df.columns:
            return c
    raise KeyError(f"Need one of {[main]+aliases} in columns: {list(df.columns)[:8]} ...")

# ---------- discovery ----------
def list_stems() -> List[str]:
    """Find stems by scanning files named '<stem>_rodrigues.csv'."""
    return sorted([p.stem.replace("_rodrigues", "") for p in RODR_DIR.glob("*_rodrigues.csv")])

# ---------- feature loaders ----------
def load_rodrigues(stem: str) -> Tuple[np.ndarray, List[int]]:
    """
    Expected columns: frame|frame_index|frame_idx, hand (0/1), then numeric Rodrigues features.
    Returns:
      X_hands [T, 2*D_hand] = hand0 || hand1 (zeros if a hand is missing),
      frames  list[int] of frame indices.
    """
    df = _read_csv(RODR_DIR / f"{stem}_rodrigues.csv")
    fcol = _pick_col(df, "frame", ["frame_index", "frame_idx"])
    hcol = _pick_col(df, "hand",  ["which", "side"])

    feat_cols = [c for c in df.columns if c not in (fcol, hcol)]
    frames = sorted(map(int, df[fcol].unique()))
    X = []
    for f in frames:
        sub = df[df[fcol] == f]
        h0 = sub[sub[hcol] == 0][feat_cols].to_numpy(np.float32)
        h1 = sub[sub[hcol] == 1][feat_cols].to_numpy(np.float32)
        h0v = h0[0] if len(h0) else np.zeros(len(feat_cols), np.float32)
        h1v = h1[0] if len(h1) else np.zeros(len(feat_cols), np.float32)
        X.append(np.concatenate([h0v, h1v], axis=0))
    return np.stack(X, 0), frames

def load_vertices(stem: str, frames: List[int]) -> np.ndarray:
    """
    Vertices file is flattened xyz triplets across columns (order agnostic).
    Missing file or frame rows -> zeros so shapes still align.
    """
    p = VERTICES_DIR / f"{stem}_vertices.csv"
    if not p.exists():
        return np.zeros((len(frames), 0), np.float32)

    df = pd.read_csv(p)
    fcol = _pick_col(df, "frame", ["frame_index", "frame_idx"])
    feat_cols = [c for c in df.columns if c != fcol]
    if not feat_cols:
        return np.zeros((len(frames), 0), np.float32)

    rows = {int(r[fcol]): r[feat_cols].to_numpy(np.float32) for _, r in df.iterrows()}
    D = len(feat_cols)
    X = np.zeros((len(frames), D), np.float32)
    for i, f in enumerate(frames):
        if f in rows:
            X[i] = rows[f]
    return X

def load_features(stem: str) -> Tuple[np.ndarray, List[int]]:
    """Concatenate Rodrigues and vertices features per frame."""
    X_h, frames = load_rodrigues(stem)
    X_v = load_vertices(stem, frames)
    X   = np.concatenate([X_h, X_v], axis=1)  # X_v may be width 0 if vertices absent
    return X, frames

# ---------- labels from closest_global ----------
def load_labels(stem: str, num_frames: int) -> np.ndarray:
    """
    Frames within ±5 of the 'closest' frame are positive (1); others 0.
    """
    p = GLOBAL_MIN_DIR / f"{stem}_closest_global.csv"
    if not p.exists():
        return np.zeros(num_frames, np.int64)

    df = pd.read_csv(p)
    if df.empty:
        return np.zeros(num_frames, np.int64)
    fcol = _pick_col(df, "frame", ["frame_index", "frame_idx"])
    t_star = int(df.iloc[0][fcol])

    y = np.zeros(num_frames, np.int64)
    lo, hi = max(0, t_star - 5), min(num_frames, t_star + 6)  # inclusive window
    y[lo:hi] = 1
    return y

# ---------- sequences / datasets ----------
def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int, stride: int):
    """
    Convert per-frame features into overlapping [seq_len, D] windows.
    Each window inherits the label of its last frame (causal decision).
    """
    xs, ys = [], []
    for end in range(seq_len - 1, len(X), stride):
        start = end - (seq_len - 1)
        xs.append(X[start:end+1])
        ys.append(y[end])
    Xs = torch.tensor(np.stack(xs, 0), dtype=torch.float32) if xs else torch.empty(0)
    Ys = torch.tensor(np.array(ys), dtype=torch.float32)     if ys else torch.empty(0)
    return Xs, Ys

class HandoverDataset(Dataset):
    """Yields (sequence[L,D], label∈{0,1})."""
    def __init__(self, stems: List[str], seq_len: int, stride: int):
        self.samples = []
        for s in stems:
            X, frames = load_features(s)
            y = load_labels(s, len(frames))
            xs, ys = make_sequences(X, y, seq_len, stride)
            for i in range(len(xs)):
                self.samples.append((xs[i], ys[i]))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

# ---------- splits / loaders ----------
def split_stems():
    """Deterministic 70/15/15 split by stem (video-level)."""
    stems = list_stems()
    random.seed(1337); random.shuffle(stems)
    n = len(stems); n_tr = int(0.7*n); n_val = int(0.15*n)
    return stems[:n_tr], stems[n_tr:n_tr+n_val], stems[n_tr+n_val:]

def build_loaders():
    tr, va, te = split_stems()
    mk = lambda ss, shuf: DataLoader(HandoverDataset(ss, SEQ_LEN, SEQ_STRIDE),
                                     batch_size=BATCH_SIZE, shuffle=shuf, drop_last=False)
    return mk(tr, True), mk(va, False), mk(te, False)
