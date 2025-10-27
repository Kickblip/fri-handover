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
    Supports TWO schemas:

    (A) WIDE (your file): one row per frame with columns like:
        time_sec, frame_index, hand_label_0, hand_label_1, hand_score_0, ...
        rot_vec_x_0, rot_vec_y_0, rot_vec_z_0, rot_vec_x_1, rot_vec_y_1, rot_vec_z_1, ...
        -> We concatenate all features for hand 0 and hand 1 into [hand0 || hand1].

    (B) LONG: rows like (frame, hand, <features...>) where hand ∈ {0,1}.
        -> We gather the two hands per frame; zeros if one is missing.

    Returns:
      X_hands : [T, 2*D_hand]  (zeros padded if a hand is missing or a column is absent)
      frames  : [T] list of frame indices (int)
    """
    import re

    df = _read_csv(RODR_DIR / f"{stem}_rodrigues.csv")

    # --- detect frame column
    fcol = _pick_col(df, "frame", ["frame_index", "frame_idx"])

    # --------- PATH (A): WIDE FORMAT ---------
    # Heuristic: look for any columns ending with _0 or _1 (e.g., rot_vec_x_0)
    has_wide_suffix = any(re.search(r"_([01])$", c) for c in df.columns)

    if has_wide_suffix:
        # Build a consistent feature list for each hand by base-name (strip trailing _0/_1)
        def split_suffix(col: str):
            m = re.search(r"(.*)_([01])$", col)
            return (m.group(1), int(m.group(2))) if m else (None, None)

        # Collect per-hand column sets
        hand0_cols = []
        hand1_cols = []
        base_names = set()

        for c in df.columns:
            base, h = split_suffix(c)
            if base is None:
                continue
            base_names.add(base)
            if h == 0:
                hand0_cols.append((base, c))
            elif h == 1:
                hand1_cols.append((base, c))

        # Sort by base name for deterministic ordering
        base_names = sorted(base_names)
        # Build ordered column lists (if a base is missing for a hand, we will fill zeros)
        h0_map = {b: c for (b, c) in hand0_cols}
        h1_map = {b: c for (b, c) in hand1_cols}

        frames = df[fcol].astype(int).tolist()
        X_rows = []

        for _, row in df.iterrows():
            # hand 0 vector
            h0_vec = []
            for b in base_names:
                if b in h0_map:
                    h0_vec.append(row[h0_map[b]])
                else:
                    h0_vec.append(0.0)
            # hand 1 vector
            h1_vec = []
            for b in base_names:
                if b in h1_map:
                    h1_vec.append(row[h1_map[b]])
                else:
                    h1_vec.append(0.0)

            # If there are any non-feature columns (like labels/scores) that also match _0/_1,
            # they’ll be included; that’s okay as long as they are numeric. If not numeric,
            # coerce to 0.
            h0_vec = [float(x) if pd.notna(x) else 0.0 for x in h0_vec]
            h1_vec = [float(x) if pd.notna(x) else 0.0 for x in h1_vec]

            X_rows.append(np.array(h0_vec + h1_vec, dtype=np.float32))

        X = np.stack(X_rows, axis=0) if len(X_rows) else np.zeros((0, 0), np.float32)
        return X, frames

    # --------- PATH (B): LONG FORMAT ---------
    # Expect 'hand' column and one row per (frame, hand).
    hcol = _pick_col(df, "hand", ["which", "side"])
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
