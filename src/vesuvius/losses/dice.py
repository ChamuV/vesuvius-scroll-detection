# src/vesuvius/losses/dice.py
from __future__ import annotations

import torch


def dice_loss_from_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    Binary Dice loss computed from logits.

    logits:  (B, 1, D, H, W)
    targets: (B, 1, D, H, W) float in {0,1}

    Returns a scalar loss.
    """
    probs = torch.sigmoid(logits)

    dims = (2, 3, 4) # spatial dims
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()