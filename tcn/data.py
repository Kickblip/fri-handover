"""
Data utilities for TCN (Aligned with Team's Logic).
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

# --- IMPORTS ---
import sys
import os
sys.path.append(str(Path(__file__).resolve().parent))

from tcn_config import (HANDS_DIR, BOX_DIR, WORLD_DIR, VERTICES_DIR,
                        SEQ_LEN, SEQ_STRIDE, BATCH_SIZE, FUTURE_FRAMES,
                        TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists(): raise FileNotFoundError(f"Missing CSV: {p}")
    return pd.read_csv(p)

def _pick_col(df: pd.DataFrame, main: str, aliases: List[str]) -> str:
    for c in [main] + aliases:
        if c in df.columns: return c
    raise KeyError(f"Need one of {[main]+aliases} in columns: {list(df.columns)[:8]} ...")

def list_stems() -> List[str]:
    stems = []
    if HANDS_DIR.exists():
        found_files = list(HANDS_DIR.glob("*_video_hands.csv"))
        for p in found_files:
            stem = p.stem.replace("_hands", "")
            stems.append(stem)
    return sorted(stems)

def _ensure_hand_consistency(X0: np.ndarray, X1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure hand0 and hand1 remain consistent throughout the video."""
    T = X0.shape[0]
    if T < 2: return X0, X1
    
    h0_reshaped = X0.reshape(T, 21, 3)
    h1_reshaped = X1.reshape(T, 21, 3)
    X0_corrected = X0.copy()
    X1_corrected = X1.copy()
    h0_corrected = h0_reshaped.copy()
    h1_corrected = h1_reshaped.copy()
    
    key_landmarks = [0, 9, 4]
    max_movement_per_frame = 0.15
    
    for t in range(1, T):
        prev_h0 = h0_corrected[t-1]
        prev_h1 = h1_corrected[t-1]
        curr_h0 = h0_reshaped[t]
        curr_h1 = h1_reshaped[t]
        
        if np.allclose(curr_h0, 0, atol=0.01) or np.allclose(curr_h1, 0, atol=0.01): continue
        
        dist_h0_to_prev_h0 = 0.0
        dist_h0_to_prev_h1 = 0.0
        dist_h1_to_prev_h0 = 0.0
        dist_h1_to_prev_h1 = 0.0
        
        valid = 0
        for lm_idx in key_landmarks:
            if lm_idx >= 21: continue
            valid += 1
            dist_h0_to_prev_h0 += np.linalg.norm(curr_h0[lm_idx] - prev_h0[lm_idx])
            dist_h0_to_prev_h1 += np.linalg.norm(curr_h0[lm_idx] - prev_h1[lm_idx])
            dist_h1_to_prev_h0 += np.linalg.norm(curr_h1[lm_idx] - prev_h0[lm_idx])
            dist_h1_to_prev_h1 += np.linalg.norm(curr_h1[lm_idx] - prev_h1[lm_idx])
        if valid == 0: continue
        
        dist_h0_to_prev_h0 /= valid
        dist_h0_to_prev_h1 /= valid
        dist_h1_to_prev_h0 /= valid
        dist_h1_to_prev_h1 /= valid

        h0_should_be_h0 = dist_h0_to_prev_h0 < dist_h0_to_prev_h1
        h1_should_be_h1 = dist_h1_to_prev_h1 < dist_h1_to_prev_h0
        
        switched = False
        if not h0_should_be_h0 and not h1_should_be_h1:
            ratio = (dist_h0_to_prev_h0 + dist_h1_to_prev_h1) / (dist_h0_to_prev_h1 + dist_h1_to_prev_h0 + 1e-6)
            if ratio > 1.5:
                if (dist_h0_to_prev_h1 < max_movement_per_frame and dist_h1_to_prev_h0 < max_movement_per_frame):
                    switched = True
        
        if switched:
            X0_corrected[t] = X1[t].copy()
            X1_corrected[t] = X0[t].copy()
            h0_corrected[t] = h1_reshaped[t].copy()
            h1_corrected[t] = h0_reshaped[t].copy()
            
    return X0_corrected, X1_corrected

def load_both_hands_world(stem: str) -> Tuple[np.ndarray, List[int]]:
    p = HANDS_DIR / f"{stem}_hands.csv"
    if not p.exists(): raise FileNotFoundError(f"Missing hands CSV: {p}")
    df = _read_csv(p)
    fcol = _pick_col(df, "frame_idx", ["frame_index", "frame"])
    frames = df[fcol].astype(int).tolist()
    
    hand0_cols = sorted([c for c in df.columns if c.startswith("h0_")])
    hand1_cols = sorted([c for c in df.columns if c.startswith("h1_")])
    
    X0 = df[hand0_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(np.float32)
    X1 = df[hand1_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(np.float32)
    
    # Apply consistency fix
    X0, X1 = _ensure_hand_consistency(X0, X1)
    
    X = np.concatenate([X0, X1], axis=1) 
    return X, frames

def load_box_coordinates(stem: str, frames: List[int]) -> np.ndarray:
    p = BOX_DIR / f"{stem}_box.csv"
    if not p.exists(): return np.zeros((len(frames), 0), np.float32)
    df = _read_csv(p)
    vertex_cols = sorted([c for c in df.columns if re.match(r"v\d+_[xyz]$", c)])
    if not vertex_cols: return np.zeros((len(frames), 0), np.float32)
    return df[vertex_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(np.float32)

def load_features(stem: str) -> Tuple[np.ndarray, List[int]]:
    X_h, frames = load_both_hands_world(stem)
    X_b = load_box_coordinates(stem, frames)
    min_len = min(len(X_h), len(X_b))
    X = np.concatenate([X_h[:min_len], X_b[:min_len]], axis=1)
    return X, frames[:min_len]

# --- CRITICAL FIX: HAND 0 (0:63) IS RECEIVER ---
def load_receiving_hand_world(stem: str) -> Tuple[np.ndarray, List[int]]:
    X_both, frames = load_both_hands_world(stem)
    # TEAM'S LOGIC: Hand 0 is the Receiver
    X_target = X_both[:, 0:63]
    return X_target, frames

def load_giving_hand_world(stem: str) -> Tuple[np.ndarray, List[int]]:
    X_both, frames = load_both_hands_world(stem)
    # TEAM'S LOGIC: Hand 1 is the Giver
    X_giving = X_both[:, 63:126]
    return X_giving, frames
# -----------------------------------------------

def make_sequences_with_targets(X_input, X_target, frames, seq_len, stride, future_frames):
    xs, ys = [], []
    min_len = min(len(X_input), len(X_target))
    X_input = X_input[:min_len]; X_target = X_target[:min_len]
    valid_range = range(seq_len - 1, len(X_input) - future_frames, stride)
    for end in valid_range:
        start = end - (seq_len - 1)
        x_seq = X_input[start:end+1]
        y_future = X_target[end+1 : end+1+future_frames]
        xs.append(x_seq); ys.append(y_future)
    if not xs: return torch.empty(0), torch.empty(0)
    return torch.tensor(np.stack(xs, 0), dtype=torch.float32), torch.tensor(np.stack(ys, 0), dtype=torch.float32)

class HandoverDataset(Dataset):
    def __init__(self, stems, seq_len, stride, future_frames):
        self.samples = []
        for s in stems:
            try:
                X_in, frames_in = load_features(s)
                X_tgt, _ = load_receiving_hand_world(s)
                xs, ys = make_sequences_with_targets(X_in, X_tgt, frames_in, seq_len, stride, future_frames)
                for i in range(len(xs)): self.samples.append((xs[i], ys[i]))
            except Exception: continue
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def list_stems() -> List[str]:
    stems = []
    if HANDS_DIR.exists():
        found = list(HANDS_DIR.glob("*_video_hands.csv"))
        for p in found: stems.append(p.stem.replace("_hands", ""))
    return sorted(stems)

def split_stems(stems_to_use=None):
    stems = list_stems() if stems_to_use is None else stems_to_use
    random.seed(1337)
    random.shuffle(stems)
    n = len(stems)
    n_train = max(1, int(n * TRAIN_SPLIT))
    n_val = max(1, int(n * VAL_SPLIT))
    return stems[:n_train], stems[n_train:n_train+n_val], stems[n_train+n_val:]

def build_loaders(stems_to_use=None):
    tr, va, te = split_stems(stems_to_use)
    def mk(ss, shuf):
        if not ss: return None
        ds = HandoverDataset(ss, SEQ_LEN, SEQ_STRIDE, FUTURE_FRAMES)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuf) if len(ds) > 0 else None
    return mk(tr, True), mk(va, False), mk(te, False)