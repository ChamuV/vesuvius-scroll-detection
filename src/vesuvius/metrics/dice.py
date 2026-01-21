# src/vesuvius/metrics/dice.py
from __future__ import annotations

import torch


def dice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> dict[str, float]:
    """
    Dice score for binary segmentation computed from logits, with optional valid mask.

    P = 1{ sigmoid(logits) >= threshold }
    G = targets in {0,1}

    If valid is provided, voxels where valid==0 are excluded from both P and G.

    Args:
        logits:  (B, 1, D, H, W)
        targets: (B, 1, D, H, W) in {0,1}
        valid:   (B, 1, D, H, W) bool or 0/1 mask (optional)
        threshold: probability threshold
        eps: numerical stability

    Returns:
        {"dice": float}
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    if valid is not None:
        v = valid.float()
        preds = preds * v
        targets = targets * v

    dims = (2, 3, 4)
    intersection = (preds * targets).sum(dim=dims)
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (denom + eps)
    return {"dice": float(dice.mean().item())}