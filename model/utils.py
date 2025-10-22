"""
Utility functions for computing metrics and evaluation.
"""

import torch

@torch.no_grad()
def binary_metrics(logits, y, thresh=0.0):
    """
    Compute precision, recall, F1, and accuracy.
    """
    preds = (logits >= thresh).float()
    tp = (preds * y).sum()
    fp = (preds * (1 - y)).sum()
    fn = ((1 - preds) * y).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (preds == y).float().mean()
    return dict(p=float(prec), r=float(rec), f1=float(f1), acc=float(acc))
