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
from .config import (WORLD_DIR, VERTICES_DIR,
                     SEQ_LEN, SEQ_STRIDE, BATCH_SIZE, FUTURE_FRAMES)

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
    """Find stems by scanning files named '<stem>_world.csv'."""
    return sorted([p.stem.replace("_world", "") for p in WORLD_DIR.glob("*_world.csv")])

# ---------- feature loaders ----------
def load_both_hands_world(stem: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load world coordinates (x, y, z) for both hands (hand_0 and hand_1).
    Returns concatenated features: [hand_0 world coords || hand_1 world coords].
    
    Expected CSV format: columns ending with:
    - '_world_x_0', '_world_y_0', '_world_z_0' for hand 0
    - '_world_x_1', '_world_y_1', '_world_z_1' for hand 1
    
    Returns:
      X_hands : [T, 126]  (63 features per hand: 21 landmarks × 3 coords)
      frames  : [T] list of frame indices (int)
    """
    p = WORLD_DIR / f"{stem}_world.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing world CSV: {p}")
    
    df = _read_csv(p)
    fcol = _pick_col(df, "frame", ["frame_index", "frame_idx"])
    frames = df[fcol].astype(int).tolist()
    
    # Extract hand_0 world coordinate columns (ending with _0)
    hand0_cols = [c for c in df.columns if c.endswith("_world_x_0") or 
                  c.endswith("_world_y_0") or c.endswith("_world_z_0")]
    
    if not hand0_cols:
        # Fallback: try to find any columns with _0 suffix and x/y/z pattern
        hand0_cols = [c for c in df.columns if re.search(r"_world_[xyz]_0$", c)]
    
    # Extract hand_1 world coordinate columns (ending with _1)
    hand1_cols = [c for c in df.columns if c.endswith("_world_x_1") or 
                  c.endswith("_world_y_1") or c.endswith("_world_z_1")]
    
    if not hand1_cols:
        # Fallback: try to find any columns with _1 suffix and x/y/z pattern
        hand1_cols = [c for c in df.columns if re.search(r"_world_[xyz]_1$", c)]
    
    if not hand0_cols:
        all_cols = list(df.columns)
        print(f"Available columns in {p.name}: {all_cols[:10]}... (showing first 10)")
        raise ValueError(f"No hand_0 world coordinates found in {p}. Expected columns ending with '_world_x_0', '_world_y_0', or '_world_z_0'")
    
    if not hand1_cols:
        all_cols = list(df.columns)
        print(f"Available columns in {p.name}: {all_cols[:10]}... (showing first 10)")
        raise ValueError(f"No hand_1 world coordinates found in {p}. Expected columns ending with '_world_x_1', '_world_y_1', or '_world_z_1'")
    
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
    
    print(f"    Loaded world coordinates: hand_0 shape {X0.shape}, hand_1 shape {X1.shape}, combined {X.shape}")
    return X, frames

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
    """Load world coordinates for both hands per frame (input features)."""
    X_h, frames = load_both_hands_world(stem)
    X_v = load_vertices(stem, frames)
    X   = np.concatenate([X_h, X_v], axis=1)  # X_v may be width 0 if vertices absent
    return X, frames

# ---------- receiving hand (target) loader ----------
def load_receiving_hand_world(stem: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load receiving hand (hand_1) world coordinates as targets.
    Returns world coordinates for all 21 landmarks (63 features: x,y,z for each).
    """
    p = WORLD_DIR / f"{stem}_world.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing world CSV: {p}")
    
    df = _read_csv(p)
    fcol = _pick_col(df, "frame", ["frame_index", "frame_idx"])
    frames = df[fcol].astype(int).tolist()
    
    # Extract all hand_1 world coordinate columns (ending with _1)
    hand1_cols = [c for c in df.columns if c.endswith("_world_x_1") or 
                  c.endswith("_world_y_1") or c.endswith("_world_z_1")]
    
    if not hand1_cols:
        # Fallback: try to find any columns with _1 suffix and x/y/z pattern
        hand1_cols = [c for c in df.columns if re.search(r"_world_[xyz]_1$", c)]
    
    if not hand1_cols:
        # Last resort: any column with _1 that looks like coordinates
        all_cols = list(df.columns)
        print(f"Available columns in {p.name}: {all_cols[:10]}... (showing first 10)")
        raise ValueError(f"No hand_1 world coordinates found in {p}. Expected columns ending with '_world_x_1', '_world_y_1', or '_world_z_1'")
    
    # Sort columns to ensure consistent ordering (x, y, z for each landmark)
    hand1_cols = sorted(hand1_cols)
    
    # Extract features - convert to numeric, coercing errors to NaN
    # This handles any non-numeric values (strings, empty cells) by converting them to NaN
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
        print(f"      WARNING: Only {len(X_input)} frames available, need at least {min_required} (seq_len={seq_len} + future_frames={future_frames})")
        return torch.empty(0), torch.empty(0)
    
    valid_range = range(seq_len - 1, len(X_input) - future_frames, stride)
    print(f"      Creating sequences: range({seq_len - 1}, {len(X_input) - future_frames}, {stride}) = {list(valid_range)[:5]}... (showing first 5)")
    
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
        print(f"      WARNING: No sequences created (valid range was empty)")
        return torch.empty(0), torch.empty(0)
    
    print(f"      Created {len(xs)} sequences")
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
                print(f"  Loading {s}...")
                X_input, frames_input = load_features(s)
                print(f"    Input features shape: {X_input.shape}, frames: {len(frames_input)}")
                
                # Load target features (receiving hand world coordinates)
                X_target, frames_target = load_receiving_hand_world(s)
                print(f"    Target features shape: {X_target.shape}, frames: {len(frames_target)}")
                
                # Align frames - use common frames
                # Simple approach: use minimum length
                min_len = min(len(X_input), len(X_target))
                print(f"    Aligning to {min_len} frames (min of input={len(X_input)}, target={len(X_target)})")
                X_input = X_input[:min_len]
                X_target = X_target[:min_len]
                frames = frames_input[:min_len]
                
                # Create sequences
                xs, ys = make_sequences_with_targets(
                    X_input, X_target, frames, seq_len, stride, future_frames
                )
                print(f"    Created {len(xs)} sequences from {min_len} frames")
                
                if len(xs) == 0:
                    print(f"    WARNING: No valid sequences created for {s} (need at least {seq_len + future_frames} = {seq_len + future_frames} frames, have {min_len})")
                    continue
                
                for i in range(len(xs)):
                    self.samples.append((xs[i], ys[i]))
                print(f"    Added {len(xs)} samples from {s}")
            except Exception as e:
                import traceback
                print(f"Warning: Skipping {s} due to error: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
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
    Deterministic split by stem (video-level).
    If stems_to_use is provided, only uses those stems.
    With only 2 stems: puts both in training (no validation/test split).
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
    
    random.seed(1337); random.shuffle(stems)
    
    # With only 2 stems, put both in training
    if len(stems) <= 2:
        return stems, [], []  # All training, no val/test
    
    # Otherwise use 70/15/15 split
    n = len(stems); n_tr = int(0.7*n); n_val = int(0.15*n)
    return stems[:n_tr], stems[n_tr:n_tr+n_val], stems[n_tr+n_val:]

def build_loaders(stems_to_use: Optional[List[str]] = None):
    """
    Build data loaders.
    If stems_to_use is provided, only uses those stems for training.
    """
    print(f"Building loaders with stems_to_use={stems_to_use}")
    tr, va, te = split_stems(stems_to_use)
    print(f"Split result: train={len(tr)} stems, val={len(va)} stems, test={len(te)} stems")
    
    def mk(ss, shuf):
        """Create DataLoader, handling empty datasets."""
        if not ss:
            print(f"  No stems provided for this split")
            return None
        
        print(f"  Creating dataset from {len(ss)} stems: {ss}")
        try:
            dataset = HandoverDataset(ss, SEQ_LEN, SEQ_STRIDE, FUTURE_FRAMES)
            if len(dataset) == 0:
                print(f"  Dataset created but has 0 samples")
                return None
            print(f"  Dataset created with {len(dataset)} samples")
            return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuf, drop_last=False)
        except RuntimeError as e:
            print(f"  Error creating dataset: {e}")
            import traceback
            print(f"  Full traceback:\n{traceback.format_exc()}")
            return None
        except Exception as e:
            print(f"  Unexpected error: {e}")
            import traceback
            print(f"  Full traceback:\n{traceback.format_exc()}")
            return None
    
    print("\nCreating train loader...")
    train_ld = mk(tr, True)
    print("\nCreating val loader...")
    val_ld = mk(va, False)
    print("\nCreating test loader...")
    test_ld = mk(te, False)
    
    if train_ld is None:
        raise RuntimeError("Train dataset is empty. Check that data files exist and have enough frames.")
    
    return train_ld, val_ld, test_ld
