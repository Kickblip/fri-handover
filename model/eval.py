"""
Validation script:
- Frame-level: sweep decision thresholds on VAL to maximize F1, then report VAL/TEST metrics at that threshold.
- Event-level: compare the predicted peak probability time (argmax) against the closest_global 'frame' using Hit@k.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .config import SEQ_LEN, GLOBAL_MIN_DIR
from .data import build_loaders, list_stems, load_features, split_stems
from .model import HandoverTransformer
from .utils import set_seed, binary_metrics_from_logits

def load_checkpoint(path: Path, device: str):
    """Restore a checkpoint and model config."""
    ckpt = torch.load(path, map_location=device)
    model = HandoverTransformer(
        in_dim=ckpt["in_dim"],
        d_model=ckpt["cfg"]["D_MODEL"],
        nhead=ckpt["cfg"]["N_HEAD"],
        nlayers=ckpt["cfg"]["N_LAYERS"],
        ffdim=ckpt["cfg"]["FFN_DIM"],
        dropout=ckpt["cfg"]["DROPOUT"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["cfg"].get("SEQ_LEN", SEQ_LEN)

@torch.no_grad()
def logits_over(loader, model, device):
    """
    Collect logits and labels over an entire DataLoader (for frame-level metrics).
    """
    logits, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits.append(model(x).cpu().numpy())
        labels.append(y.cpu().numpy())
    if not logits:
        return np.array([]), np.array([])
    return np.concatenate(logits), np.concatenate(labels)

def sweep_thresholds(logits: np.ndarray, labels: np.ndarray):
    """
    Choose the logit threshold that maximizes F1 on validation.
    We sweep probability grid in [0.1..0.9], convert to logits, and evaluate.
    """
    probs = np.linspace(0.1, 0.9, 33)
    grid = np.log(probs) - np.log(1 - probs)  # prob->logit
    best_f1, best_th = -1.0, 0.0
    for th in grid:
        pred = (logits >= th).astype(np.float32)
        tp = (pred * labels).sum(); fp = (pred * (1 - labels)).sum(); fn = ((1 - pred) * labels).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return dict(f1=best_f1, thresh=best_th, prob=1 / (1 + np.exp(-best_th)))

@torch.no_grad()
def per_frame_probs(stem: str, model, seq_len: int, device: str):
    """
    Produce per-frame probabilities for a single stem (argmax used for event time).
    """
    X, frames = load_features(stem)
    T = X.shape[0]
    logits = np.full((T,), np.nan, dtype=np.float32)
    for end in range(seq_len - 1, T):
        start = end - (seq_len - 1)
        chunk = torch.from_numpy(X[start:end+1]).unsqueeze(0).float().to(device)
        logits[end] = model(chunk).item()
    logits[:seq_len - 1] = logits[seq_len - 1]
    probs = 1.0 / (1.0 + np.exp(-logits))
    return frames, probs

def load_true_event_time(stem: str):
    """
    Read the closest_global 'frame' (proxy event time).
    """
    p = GLOBAL_MIN_DIR / f"{stem}_closest_global.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "frame" not in df.columns or len(df) == 0:
        return None
    return int(df.iloc[0]["frame"])

def main():
    set_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load splits and restore best model
    train_ld, val_ld, test_ld = build_loaders()
    stems = list_stems()
    if not stems:
        raise RuntimeError("No stems found for evaluation.")
    model, seq_len = load_checkpoint(Path("handover_best.pt"), device)

    # ---------- Frame-level: pick threshold on VAL ----------
    v_logits, v_labels = logits_over(val_ld, model, device)
    if v_logits.size:
        best = sweep_thresholds(v_logits, v_labels)
        print(f"VAL: best F1={best['f1']:.3f} at logit={best['thresh']:.3f} (~prob={best['prob']:.2f})")
        v_metrics = binary_metrics_from_logits(torch.from_numpy(v_logits),
                                               torch.from_numpy(v_labels),
                                               thresh=best["thresh"])
        print(f"VAL metrics: F1={v_metrics['f1']:.3f} | P={v_metrics['precision']:.3f} | "
              f"R={v_metrics['recall']:.3f} | Acc={v_metrics['acc']:.3f}")
        chosen_thresh = best["thresh"]
    else:
        print("No VAL split found; defaulting threshold to 0.0 (prob 0.5).")
        chosen_thresh = 0.0

    # ---------- Frame-level: report TEST at chosen threshold ----------
    t_logits, t_labels = logits_over(test_ld, model, device)
    if t_logits.size:
        t_metrics = binary_metrics_from_logits(torch.from_numpy(t_logits),
                                               torch.from_numpy(t_labels),
                                               thresh=chosen_thresh)
        print(f"TEST metrics: F1={t_metrics['f1']:.3f} | P={t_metrics['precision']:.3f} | "
              f"R={t_metrics['recall']:.3f} | Acc={t_metrics['acc']:.3f}")
    else:
        print("No TEST split for frame-level evaluation.")

    # ---------- Event-level: Hit@k using peak prob vs closest_global ----------
    tr, va, te = split_stems()
    for name, stems_split in [("VAL", va), ("TEST", te)]:
        if not stems_split:
            print(f"{name}: no stems for event-level evaluation.")
            continue
        hits3 = hits5 = hits10 = total = 0
        for s in stems_split:
            frames, probs = per_frame_probs(s, model, seq_len, device)
            pred_t = int(np.nanargmax(probs))  # predicted event time = probability peak
            true_t = load_true_event_time(s)
            if true_t is None:
                continue
            hits3  += int(abs(pred_t - true_t) <= 3)
            hits5  += int(abs(pred_t - true_t) <= 5)
            hits10 += int(abs(pred_t - true_t) <= 10)
            total  += 1
        if total:
            print(f"{name} Hit@3={hits3/total:.3f} ({hits3}/{total}) | "
                  f"Hit@5={hits5/total:.3f} ({hits5}/{total}) | "
                  f"Hit@10={hits10/total:.3f} ({hits10}/{total})")

if __name__ == "__main__":
    main()
