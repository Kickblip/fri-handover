"""
Train the Handover Transformer using weak labels:
- Reads features (Rodrigues + vertices), windows them, and trains a small Transformer.
- Uses BCEWithLogitsLoss with a positive class weight (to fight imbalance).
- Monitors validation F1 and early-stops if no improvement.
- Saves the best checkpoint to 'handover_best.pt'.
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import *
from .data import build_loaders, list_stems, load_features
from .model import HandoverTransformer
from .utils import set_seed, binary_metrics_from_logits

def train():
    # 1) Reproducibility + device selection
    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🟢 Using device: {device}")

    # 2) Data
    train_ld, val_ld, test_ld = build_loaders()

    # 3) Dynamically infer input dimension from any one stem (safe if all stems share schema)
    stems = list_stems()
    if not stems:
        raise RuntimeError("No stems found in the Rodrigues folder.")
    X_sample, _ = load_features(stems[0])
    in_dim = X_sample.shape[1]
    print(f"Detected input feature dimension: {in_dim}")

    # 4) Model / loss / optimizer / scheduler
    model = HandoverTransformer(in_dim, D_MODEL, N_HEAD, N_LAYERS, FFN_DIM, DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_CLASS_WEIGHT], device=device))
    opt = AdamW(model.parameters(), lr=LR)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, verbose=True)

    # 5) Train with early stopping on validation F1
    best_f1 = 0.0
    epochs_no_improve = 0
    ckpt_path = Path("handover_best.pt")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ----- training epoch -----
        model.train()
        total_loss = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stabilize training
            opt.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / max(1, len(train_ld.dataset))

        # ----- validation -----
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                all_logits.append(model(x).cpu())
                all_y.append(y.cpu())

        if all_logits:
            logits_cat = torch.cat(all_logits, dim=0)
            y_cat = torch.cat(all_y, dim=0)
            metrics = binary_metrics_from_logits(logits_cat, y_cat, thresh=0.0)  # logit 0 == prob 0.5
            f1 = metrics["f1"]
            print(f"Epoch {epoch:02d} | TrainLoss {train_loss:.4f} | "
                  f"Val F1 {f1:.3f} (P {metrics['precision']:.3f}, R {metrics['recall']:.3f}, Acc {metrics['acc']:.3f})")
            # step scheduler on F1 (maximize)
            sched.step(f1)

            # checkpoint the best model
            if f1 > best_f1:
                best_f1 = f1
                epochs_no_improve = 0
                torch.save({
                    "model": model.state_dict(),
                    "in_dim": in_dim,
                    "cfg": dict(D_MODEL=D_MODEL, N_HEAD=N_HEAD, N_LAYERS=N_LAYERS,
                                FFN_DIM=FFN_DIM, DROPOUT=DROPOUT, SEQ_LEN=SEQ_LEN)
                }, ckpt_path)
                print(f"✅ New best F1 {best_f1:.3f}; saved to {ckpt_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= EARLY_STOP_PATIENCE:
                    print("⏹ Early stopping — no validation F1 improvement.")
                    break
        else:
            # Tiny datasets might not produce a val split; that's fine.
            print(f"Epoch {epoch:02d} | TrainLoss {train_loss:.4f} | (no validation set)")

    # ----- optional test set summary (if present) -----
    if test_ld.dataset:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for x, y in test_ld:
                x, y = x.to(device), y.to(device)
                all_logits.append(model(x).cpu()); all_y.append(y.cpu())
        if all_logits:
            logits = torch.cat(all_logits); ys = torch.cat(all_y)
            m = binary_metrics_from_logits(logits, ys)
            print(f"TEST — F1 {m['f1']:.3f} | P {m['precision']:.3f} | R {m['recall']:.3f} | Acc {m['acc']:.3f}")

if __name__ == "__main__":
    train()
