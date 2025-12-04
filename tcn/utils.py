"""
Small helpers for reproducibility and binary classification metrics.
"""

from __future__ import annotations
import random
import numpy as np
import torch

def set_seed(seed: int = 1337):
    """
    Set random seeds across Python/NumPy/PyTorch for repeatable runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def binary_metrics_from_logits(logits: torch.Tensor, y_true: torch.Tensor, thresh: float = 0.0):
    """
    Compute precision/recall/F1/accuracy from raw logits and ground-truth labels.

    Args:
      logits : [N] tensor of raw scores (pre-sigmoid)
      y_true : [N] tensor of {0,1} labels
      thresh : decision threshold on logits; 0.0 corresponds to prob=0.5

    Returns:
      dict with precision, recall, f1, acc
    """
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
