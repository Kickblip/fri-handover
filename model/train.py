"""
Train the transformer using labels derived from closest_global:
- BCEWithLogitsLoss + positive class weighting.
- Early stopping on validation F1.
- Best model saved to dataset/model_output/checkpoints/handover_transformer.pt
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .config import (D_MODEL, N_HEAD, N_LAYERS, FFN_DIM, DROPOUT,
                     LR, MAX_EPOCHS, POS_CLASS_WEIGHT, EARLY_STOP_PATIENCE,
                     CKPT_PATH)
from .data import build_loaders, list_stems, load_features
from .model import HandoverTransformer

# ----- small utils -----
def set_seed(seed: int = 1337):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def binary_metrics_from_logits(logits: torch.Tensor, y_true: torch.Tensor, thresh: float = 0.0):
    y_pred = (logits >= thresh).float()
    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()
    tn = ((1 - y_pred) * (1 - y_true)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    return dict(precision=float(prec), recall=float(rec), f1=float(f1), acc=float(acc))

# ----- training loop -----
def train():
    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🟢 Device: {device}")

    train_ld, val_ld, test_ld = build_loaders()

    stems = list_stems()
    if not stems:
        raise RuntimeError("No stems discovered in Rodrigues folder.")
    in_dim = load_features(stems[0])[0].shape[1]
    print(f"Detected input dim: {in_dim}")

    model = HandoverTransformer(in_dim, D_MODEL, N_HEAD, N_LAYERS, FFN_DIM, DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_CLASS_WEIGHT], device=device))
    opt = AdamW(model.parameters(), lr=LR)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, verbose=True)

    best_f1, stale = 0.0, 0
    for epoch in range(1, MAX_EPOCHS+1):
        # train
        model.train()
        total, n = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * x.size(0); n += x.size(0)
        tr_loss = total / max(1, n)

        # validate
        model.eval()
        logits_all, y_all = [], []
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                logits_all.append(model(x).cpu()); y_all.append(y.cpu())

        if logits_all:
            logits = torch.cat(logits_all); ys = torch.cat(y_all)
            m = binary_metrics_from_logits(logits, ys, thresh=0.0)
            sched.step(m["f1"])
            print(f"Epoch {epoch:02d} | TrainLoss {tr_loss:.4f} | "
                  f"VAL F1 {m['f1']:.3f} (P {m['precision']:.3f}, R {m['recall']:.3f}, Acc {m['acc']:.3f})")

            if m["f1"] > best_f1:
                best_f1, stale = m["f1"], 0
                torch.save({
                    "model": model.state_dict(),
                    "in_dim": in_dim,
                    "cfg": dict(D_MODEL=D_MODEL, N_HEAD=N_HEAD, N_LAYERS=N_LAYERS,
                                FFN_DIM=FFN_DIM, DROPOUT=DROPOUT)
                }, CKPT_PATH)
                print(f"✅ New best F1 {best_f1:.3f} — saved {CKPT_PATH}")
            else:
                stale += 1
                if stale >= EARLY_STOP_PATIENCE:
                    print("⏹ Early stopping (no F1 improvement).")
                    break
        else:
            print(f"Epoch {epoch:02d} | TrainLoss {tr_loss:.4f} | (no VAL set)")

    # optional TEST summary
    if test_ld:
        state = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(state["model"]); model.eval()
        logits_all, y_all = [], []
        with torch.no_grad():
            for x, y in test_ld:
                x, y = x.to(device), y.to(device)
                logits_all.append(model(x).cpu()); y_all.append(y.cpu())
        if logits_all:
            logits = torch.cat(logits_all); ys = torch.cat(y_all)
            m = binary_metrics_from_logits(logits, ys)
            print(f"TEST: F1 {m['f1']:.3f} | P {m['precision']:.3f} | R {m['recall']:.3f} | Acc {m['acc']:.3f}")

if __name__ == "__main__":
    train()
