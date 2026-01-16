# src/vesuvius/metrics/dice.py
from __future__ import annotations

import torch


def dice_from_logits(logits, targets, *, threshold: float = 0.5, eps: float = 1e-7):
    ####DOC STRING WITH FORMULA
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    dims = (2, 3, 4)
    inter = (preds * targets).sum(dims)
    denom = preds.sum(dims) + targets.sum(dims)

    return {"dice": ((2 * inter + eps) / (denom + eps)).mean().item()}