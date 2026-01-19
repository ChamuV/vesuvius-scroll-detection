# src/vesuvius/losses/dice.py
from __future__ import annotations

import torch


def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Binary Dice loss from logits with optional mask.

    logits:  (B,1,D,H,W)
    targets: (B,1,D,H,W) float in {0,1}
    valid:   optional bool/0-1 mask (B,1,D,H,W); ignored voxels excluded
    """
    probs = torch.sigmoid(logits)

    if valid is not None:
        v = valid.float()
        probs = probs * v
        targets = targets * v

    dims = (2, 3, 4)
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()