"""
Main training script for the Handover Transformer.

It:
- Loads sequences and proxy labels
- Trains the model
- Tracks validation F1 score
- Saves the best-performing checkpoint
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .data import build_loaders, list_stems, load_rodrigues
from .model import HandoverTransformer
from .utils import binary_metrics
from .config import *

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🟢 Using device: {device}")

    # 1. Build data loaders
    train_ld, val_ld, test_ld = build_loaders()

    # 2. Determine feature dimension dynamically from first file
    sample_stem = list_stems()[0]
    X, _ = load_rodrigues(sample_stem)
    in_dim = X.shape[1]

    # 3. Initialize model
    model = HandoverTransformer(in_dim, D_MODEL, N_HEAD, N_LAYERS, FFN_DIM, DROPOUT).to(device)

    # 4. Loss, optimizer, and scheduler
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_CLASS_WEIGHT], device=device))
    opt = AdamW(model.parameters(), lr=LR)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, verbose=True)

    # 5. Training loop
    best_f1, no_imp = 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0

        # ----- Training phase -----
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(x)
        train_loss = total_loss / len(train_ld.dataset)

        # ----- Validation phase -----
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                all_logits.append(model(x).cpu())
                all_y.append(y.cpu())

        if not all_logits:
            continue

        logits = torch.cat(all_logits)
        ys = torch.cat(all_y)
        m = binary_metrics(logits, ys)

        print(f"Epoch {epoch:02d} | TrainLoss {train_loss:.4f} | "
              f"F1 {m['f1']:.3f} | P {m['p']:.3f} | R {m['r']:.3f}")

        sched.step(m["f1"])

        # Save best model by F1
        if m["f1"] > best_f1:
            best_f1, no_imp = m["f1"], 0
            torch.save({
                "model": model.state_dict(),
                "in_dim": in_dim,
                "cfg": dict(D_MODEL=D_MODEL, N_HEAD=N_HEAD, N_LAYERS=N_LAYERS,
                            FFN_DIM=FFN_DIM, DROPOUT=DROPOUT)
            }, "handover_best.pt")
            print("✅ Saved new best model.")
        else:
            no_imp += 1
            if no_imp >= EARLY_STOP_PATIENCE:
                print("⏹ Early stop triggered.")
                break

if __name__ == "__main__":
    train()
