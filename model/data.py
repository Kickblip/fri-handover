"""
Data utilities:
- Load per-frame input features: World coordinates (x,y,z) for both hands concatenated.
- Load receiving hand (hand_1) world coordinates as targets.
- Convert time-series into fixed-length windows with future frame targets.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import random
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from .config import (HANDS_DIR, BOX_DIR, WORLD_DIR, VERTICES_DIR,
                     SEQ_LEN, SEQ_STRIDE, BATCH_SIZE, FUTURE_FRAMES,
                     TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

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
    """
    Find stems by scanning files named '{number}_video_hands.csv' in the hands folder.
    Returns stems like '1_video', '2_video', etc.
    """
    stems = []
    
    # Check new format: model_dataset/handover-csv/hands/
    if HANDS_DIR.exists():
        found_files = list(HANDS_DIR.glob("*_video_hands.csv"))
        for p in found_files:
            # Extract stem: "1_video_hands.csv" -> "1_video"
            stem = p.stem.replace("_hands", "")
            stems.append(stem)
    
    # Fallback to old format ONLY if new format not found (for backward compatibility)
    if not stems and WORLD_DIR.exists():
        found_files = list(WORLD_DIR.glob("*_world.csv"))
        for p in found_files:
            stems.append(p.stem.replace("_world", ""))
    
    return sorted(stems)

# ---------- feature loaders ----------
def load_both_hands_world(stem: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load world coordinates (x, y, z) for both hands (hand_0 and hand_1).
    Returns concatenated features: [hand_0 world coords || hand_1 world coords].
    
    Expected CSV format:
    - frame_idx, h0_lm0_x, h0_lm0_y, h0_lm0_z, h0_lm1_x, ... (hand 0)
    - h1_lm0_x, h1_lm0_y, h1_lm0_z, h1_lm1_x, ... (hand 1)
    
    Returns:
      X_hands : [T, 126]  (63 features per hand: 21 landmarks × 3 coords)
      frames  : [T] list of frame indices (int)
    """
    # Try new format first
    p = HANDS_DIR / f"{stem}_hands.csv" if HANDS_DIR.exists() else None
    if p is None or not p.exists():
        # Fallback to old format
        p = WORLD_DIR / f"{stem}_world.csv"
    
    if not p.exists():
        raise FileNotFoundError(f"Missing hands CSV: {p}")
    
    df = _read_csv(p)
    fcol = _pick_col(df, "frame_idx", ["frame_index", "frame"])
    frames = df[fcol].astype(int).tolist()
    
    # Extract hand_0 columns (starting with h0_)
    hand0_cols = [c for c in df.columns if c.startswith("h0_")]
    
    # Extract hand_1 columns (starting with h1_)
    hand1_cols = [c for c in df.columns if c.startswith("h1_")]
    
    if not hand0_cols:
        raise ValueError(f"No hand_0 columns found in {p}. Expected columns starting with 'h0_' (e.g., h0_lm0_x, h0_lm0_y, h0_lm0_z)")
    
    if not hand1_cols:
        raise ValueError(f"No hand_1 columns found in {p}. Expected columns starting with 'h1_' (e.g., h1_lm0_x, h1_lm0_y, h1_lm0_z)")
    
    # Sort columns to ensure consistent ordering (x, y, z for each landmark)
    hand0_cols = sorted(hand0_cols)
    hand1_cols = sorted(hand1_cols)
    
    # Extract features - convert to numeric, coercing errors to NaN
    X0 = df[hand0_cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    X1 = df[hand1_cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    
    # Replace NaN with 0
    X0 = np.nan_to_num(X0, nan=0.0)
    X1 = np.nan_to_num(X1, nan=0.0)
    
    # Concatenate both hands: [hand_0 || hand_1]
    X = np.concatenate([X0, X1], axis=1)  # [T, 126]
    
    return X, frames

def load_box_coordinates(stem: str, frames: List[int]) -> np.ndarray:
    """
    Load AprilTag box coordinates from {stem}_box.csv.
    
    Expected CSV format: frame_idx, tag_id, coord_frame, v0_x, v0_y, v0_z, v1_x, v1_y, v1_z, ...
    Returns only the vertex coordinates (v*_x, v*_y, v*_z columns).
    
    Returns:
      X_box : [T, D_box] where D_box is the number of box coordinate features
    """
    p = BOX_DIR / f"{stem}_box.csv" if BOX_DIR.exists() else None
    if p is None or not p.exists():
        # Return empty if box file doesn't exist
        return np.zeros((len(frames), 0), np.float32)
    
    df = _read_csv(p)
    fcol = _pick_col(df, "frame_idx", ["frame_index", "frame"])
    
    # Extract vertex coordinate columns (v*_x, v*_y, v*_z)
    vertex_cols = [c for c in df.columns if re.match(r"v\d+_[xyz]$", c)]
    
    if not vertex_cols:
        return np.zeros((len(frames), 0), np.float32)
    
    # Sort columns to ensure consistent ordering
    vertex_cols = sorted(vertex_cols)
    
    # Convert to numeric with coercion
    df_numeric = df[[fcol] + vertex_cols].copy()
    df_numeric[vertex_cols] = df_numeric[vertex_cols].apply(pd.to_numeric, errors='coerce')
    
    # Build rows dictionary
    rows = {}
    for _, row in df_numeric.iterrows():
        frame_idx = int(row[fcol])
        feat_values = row[vertex_cols].to_numpy(np.float32)
        rows[frame_idx] = feat_values
    
    D = len(vertex_cols)
    X = np.zeros((len(frames), D), np.float32)
    for i, f in enumerate(frames):
        if f in rows:
            X[i] = rows[f]
    
    # Replace any remaining NaN with 0
    X = np.nan_to_num(X, nan=0.0)
    return X

def load_vertices(stem: str, frames: List[int]) -> np.ndarray:
    p = VERTICES_DIR / f"{stem}_vertices.csv"
    if not p.exists():
        return np.zeros((len(frames), 0), np.float32)

    df = pd.read_csv(p)
    fcol = _pick_col(df, "frame", ["frame_index", "frame_idx"])
    feat_cols = [c for c in df.columns if c != fcol]
    if not feat_cols:
        return np.zeros((len(frames), 0), np.float32)

    # Filter to only numeric columns and convert safely
    numeric_feat_cols = []
    for col in feat_cols:
        # Try converting to numeric - if it works, keep it
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            if not converted.isna().all():
                numeric_feat_cols.append(col)
        except (ValueError, TypeError):
            # Skip non-numeric columns
            continue
    
    if not numeric_feat_cols:
        # No numeric columns found, return empty
        return np.zeros((len(frames), 0), np.float32)
    
    # Convert to numeric with coercion
    df_numeric = df[[fcol] + numeric_feat_cols].copy()
    df_numeric[numeric_feat_cols] = df_numeric[numeric_feat_cols].apply(pd.to_numeric, errors='coerce')
    
    # Build rows dictionary
    rows = {}
    for _, row in df_numeric.iterrows():
        frame_idx = int(row[fcol])
        feat_values = row[numeric_feat_cols].to_numpy(np.float32)
        rows[frame_idx] = feat_values
    
    D = len(numeric_feat_cols)
    X = np.zeros((len(frames), D), np.float32)
    for i, f in enumerate(frames):
        if f in rows:
            X[i] = rows[f]
    
    # Replace any remaining NaN with 0
    X = np.nan_to_num(X, nan=0.0)
    return X

def load_features(stem: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load input features: both hands world coordinates + box coordinates + optional vertices.
    Returns concatenated features: [hands || box || vertices]
    """
    X_h, frames = load_both_hands_world(stem)
    X_b = load_box_coordinates(stem, frames)
    X_v = load_vertices(stem, frames)
    X   = np.concatenate([X_h, X_b, X_v], axis=1)  # X_b and X_v may be width 0 if absent
    return X, frames

# ---------- receiving hand (target) loader ----------
def load_receiving_hand_world(stem: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load receiving hand (hand_1) world coordinates as targets.
    Returns world coordinates for all 21 landmarks (63 features: x,y,z for each).
    
    Expected CSV format: frame_idx, h1_lm0_x, h1_lm0_y, h1_lm0_z, h1_lm1_x, ...
    """
    # Try new format first
    p = HANDS_DIR / f"{stem}_hands.csv" if HANDS_DIR.exists() else None
    if p is None or not p.exists():
        # Fallback to old format
        p = WORLD_DIR / f"{stem}_world.csv"
    
    if not p.exists():
        raise FileNotFoundError(f"Missing hands CSV: {p}")
    
    df = _read_csv(p)
    fcol = _pick_col(df, "frame_idx", ["frame_index", "frame"])
    frames = df[fcol].astype(int).tolist()
    
    # Extract all hand_1 columns (starting with h1_)
    hand1_cols = [c for c in df.columns if c.startswith("h1_")]
    
    if not hand1_cols:
        raise ValueError(f"No hand_1 columns found in {p}. Expected columns starting with 'h1_' (e.g., h1_lm0_x, h1_lm0_y, h1_lm0_z)")
    
    # Sort columns to ensure consistent ordering (x, y, z for each landmark)
    hand1_cols = sorted(hand1_cols)
    
    # Extract features - convert to numeric, coercing errors to NaN
    X = df[hand1_cols].apply(pd.to_numeric, errors='coerce').to_numpy(np.float32)
    
    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)
    
    return X, frames

# ---------- sequences / datasets ----------
def make_sequences_with_targets(X_input: np.ndarray, X_target: np.ndarray, 
                                  frames: List[int], seq_len: int, stride: int, 
                                  future_frames: int):
    """
    Convert per-frame features into overlapping windows with future frame targets.
    
    Args:
        X_input: [T, D_in] input features (both hands world coordinates + optional vertices)
        X_target: [T, D_out] target features (receiving hand world coords)
        frames: [T] frame indices
        seq_len: length of input sequence
        stride: stride for windows
        future_frames: number of future frames to predict
    
    Returns:
        Xs: [N, seq_len, D_in] input sequences
        Ys: [N, future_frames, D_out] target future frames
    """
    xs, ys = [], []
    
    # Align frames if needed
    if len(X_input) != len(X_target):
        # Find common frames
        min_len = min(len(X_input), len(X_target))
        X_input = X_input[:min_len]
        X_target = X_target[:min_len]
    
    # Calculate valid range for sequences
    min_required = seq_len + future_frames  # Need at least this many frames
    if len(X_input) < min_required:
        return torch.empty(0), torch.empty(0)
    
    valid_range = range(seq_len - 1, len(X_input) - future_frames, stride)
    
    for end in valid_range:
        start = end - (seq_len - 1)
        
        # Input sequence
        x_seq = X_input[start:end+1]  # [seq_len, D_in]
        
        # Target: future frames after end
        target_start = end + 1
        target_end = target_start + future_frames
        if target_end > len(X_target):
            # Pad with last frame if not enough future frames
            y_future = X_target[target_start:]
            padding = np.tile(X_target[-1:], (future_frames - len(y_future), 1))
            y_future = np.concatenate([y_future, padding], axis=0)
        else:
            y_future = X_target[target_start:target_end]  # [future_frames, D_out]
        
        xs.append(x_seq)
        ys.append(y_future)
    
    if not xs:
        return torch.empty(0), torch.empty(0)
    
    Xs = torch.tensor(np.stack(xs, 0), dtype=torch.float32)
    Ys = torch.tensor(np.stack(ys, 0), dtype=torch.float32)
    return Xs, Ys

class HandoverDataset(Dataset):
    """Yields (input_sequence[L,D_in], target_future_frames[future_frames,D_out])."""
    def __init__(self, stems: List[str], seq_len: int, stride: int, future_frames: int):
        self.samples = []
        for s in stems:
            try:
                # Load input features (both hands world coordinates + optional vertices)
                X_input, frames_input = load_features(s)
                
                # Load target features (receiving hand world coordinates)
                X_target, frames_target = load_receiving_hand_world(s)
                
                # Align frames - use common frames
                min_len = min(len(X_input), len(X_target))
                X_input = X_input[:min_len]
                X_target = X_target[:min_len]
                frames = frames_input[:min_len]
                
                # Create sequences
                xs, ys = make_sequences_with_targets(
                    X_input, X_target, frames, seq_len, stride, future_frames
                )
                
                if len(xs) == 0:
                    continue
                
                for i in range(len(xs)):
                    self.samples.append((xs[i], ys[i]))
            except Exception as e:
                continue
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples created from {len(stems)} stems. Check that data files exist and have enough frames.")

    def __len__(self):  
        return len(self.samples)
    
    def __getitem__(self, i): 
        return self.samples[i]

# ---------- splits / loaders ----------
def split_stems(stems_to_use: Optional[List[str]] = None):
    """
    Deterministic split by stem (video-level) into train/validation/test sets.
    If stems_to_use is provided, only uses those stems.
    
    Split strategy:
    - 1 stem: All in training (no val/test)
    - 2 stems: 1 train, 1 val (no test)
    - 3 stems: 1 train, 1 val, 1 test
    - 4+ stems: Uses TRAIN_SPLIT/VAL_SPLIT/TEST_SPLIT ratios
    """
    if stems_to_use is None:
        stems = list_stems()
    else:
        # Filter to only requested stems that exist
        all_stems = list_stems()
        stems = [s for s in stems_to_use if s in all_stems]
        if len(stems) < len(stems_to_use):
            missing = set(stems_to_use) - set(stems)
            print(f"Warning: Some requested stems not found: {missing}")
    
    if len(stems) == 0:
        return [], [], []
    
    # Deterministic shuffle with fixed seed
    random.seed(1337)
    stems_shuffled = stems.copy()
    random.shuffle(stems_shuffled)
    
    n = len(stems_shuffled)
    
    # Handle small datasets
    if n == 1:
        return stems_shuffled, [], []
    elif n == 2:
        return stems_shuffled[:1], stems_shuffled[1:2], []
    elif n == 3:
        return stems_shuffled[:1], stems_shuffled[1:2], stems_shuffled[2:3]
    else:
        # Use configured split ratios
        n_train = max(1, int(n * TRAIN_SPLIT))
        n_val = max(1, int(n * VAL_SPLIT))
        n_test = n - n_train - n_val  # Remaining goes to test
        
        # Ensure at least 1 in each split if possible
        if n_test == 0 and n > 2:
            n_train -= 1
            n_test = 1
        
        train_stems = stems_shuffled[:n_train]
        val_stems = stems_shuffled[n_train:n_train + n_val]
        test_stems = stems_shuffled[n_train + n_val:]
        
        return train_stems, val_stems, test_stems

def build_loaders(stems_to_use: Optional[List[str]] = None):
    """
    Build data loaders.
    If stems_to_use is provided, only uses those stems for training.
    """
    all_available_stems = list_stems()
    tr, va, te = split_stems(stems_to_use)
    
    def mk(ss, shuf):
        """Create DataLoader, handling empty datasets."""
        if not ss:
            return None
        try:
            dataset = HandoverDataset(ss, SEQ_LEN, SEQ_STRIDE, FUTURE_FRAMES)
            if len(dataset) == 0:
                return None
            return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuf, drop_last=False)
        except Exception:
            return None
    
    train_ld = mk(tr, True)
    val_ld = mk(va, False)
    test_ld = mk(te, False)
    
    if train_ld is None:
        available_stems = list_stems()
        raise RuntimeError(
            f"Train dataset is empty. "
            f"Expected files in {HANDS_DIR} (format: {{number}}_video_hands.csv). "
            f"Found {len(available_stems)} stems: {available_stems}"
        )
    
    return train_ld, val_ld, test_ld
