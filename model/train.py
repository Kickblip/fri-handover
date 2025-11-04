"""
Train the transformer to predict future frames of receiving hand:
- MSE loss for regression.
- Early stopping on validation loss.
- Best model saved to dataset/model_output/checkpoints/handover_transformer.pt
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .config import (D_MODEL, N_HEAD, N_LAYERS, FFN_DIM, DROPOUT,
                     LR, MAX_EPOCHS, EARLY_STOP_PATIENCE, FUTURE_FRAMES,
                     CKPT_PATH)
from .data import build_loaders, list_stems, load_features, load_receiving_hand_world
from .model import HandoverTransformer

# ----- small utils -----
def set_seed(seed: int = 1337):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ----- training loop -----
def train(stems_to_use: Optional[List[str]] = None):
    """
    Train the model.
    If stems_to_use is provided, only trains on those specific stems.
    """
    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🟢 Device: {device}")

    train_ld, val_ld, test_ld = build_loaders(stems_to_use)

    # Get stems used for training
    if stems_to_use is None:
        stems = list_stems()
    else:
        all_stems = list_stems()
        stems = [s for s in stems_to_use if s in all_stems]
    
    if not stems:
        raise RuntimeError("No stems available for training.")
    
    print(f"Using stems: {stems}")
    
    # Get input and output dimensions
    in_dim = load_features(stems[0])[0].shape[1]
    out_dim = load_receiving_hand_world(stems[0])[0].shape[1]
    print(f"Detected input dim: {in_dim}, output dim: {out_dim}")

    model = HandoverTransformer(
        in_dim=in_dim, out_dim=out_dim, d_model=D_MODEL, nhead=N_HEAD,
        nlayers=N_LAYERS, ffdim=FFN_DIM, dropout=DROPOUT, future_frames=FUTURE_FRAMES
    ).to(device)
    
    criterion = nn.MSELoss()
    opt = AdamW(model.parameters(), lr=LR)
    sched = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, verbose=True)

    best_val_loss, stale = float('inf'), 0
    for epoch in range(1, MAX_EPOCHS+1):
        # train
        model.train()
        total, n = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)  # x: [B, L, D_in], y: [B, future_frames, D_out]
            pred = model(x)  # [B, future_frames, D_out]
            loss = criterion(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * x.size(0); n += x.size(0)
        tr_loss = total / max(1, n)

        # validate
        model.eval()
        val_losses = []
        if val_ld is not None:
            with torch.no_grad():
                for x, y in val_ld:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = criterion(pred, y)
                    val_losses.append(loss.item())

        if val_losses:
            val_loss = sum(val_losses) / len(val_losses)
            sched.step(val_loss)
            print(f"Epoch {epoch:02d} | TrainLoss {tr_loss:.6f} | ValLoss {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss, stale = val_loss, 0
                torch.save({
                    "model": model.state_dict(),
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "cfg": dict(D_MODEL=D_MODEL, N_HEAD=N_HEAD, N_LAYERS=N_LAYERS,
                                FFN_DIM=FFN_DIM, DROPOUT=DROPOUT, FUTURE_FRAMES=FUTURE_FRAMES)
                }, CKPT_PATH)
                print(f"✅ New best ValLoss {best_val_loss:.6f} — saved {CKPT_PATH}")
            else:
                stale += 1
                if stale >= EARLY_STOP_PATIENCE:
                    print("⏹ Early stopping (no validation loss improvement).")
                    break
        else:
            print(f"Epoch {epoch:02d} | TrainLoss {tr_loss:.6f} | (no VAL set)")

    # optional TEST summary
    if test_ld is not None:
        state = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(state["model"]); model.eval()
        test_losses = []
        with torch.no_grad():
            for x, y in test_ld:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                test_losses.append(loss.item())
        if test_losses:
            test_loss = sum(test_losses) / len(test_losses)
            print(f"TEST Loss: {test_loss:.6f}")

if __name__ == "__main__":
    # Train only on these two specific files
    stems_to_use = ["1_w_b", "2_w_b"]
    train(stems_to_use=stems_to_use)
