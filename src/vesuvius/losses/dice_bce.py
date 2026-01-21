# src/vesuvius/losses/dice_bce.py
from __future__ import annotations

import torch
import torch.nn as nn

from vesuvius.losses.bce import BCELoss
from vesuvius.losses.dice import dice_loss_from_logits


class DiceBCELoss(nn.Module):
    """
    loss = w_bce * BCEWithLogits + w_dice * DiceLoss
    Supports optional valid mask (ignored voxels).
    """
    def __init__(
        self,
        w_bce: float = 0.5,
        w_dice: float = 0.5,
        eps: float = 1e-6,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.w_bce = float(w_bce)
        self.w_dice = float(w_dice)
        self.eps = float(eps)

        self.bce = BCELoss(pos_weight=pos_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce = self.bce(logits, targets, valid)
        dice = dice_loss_from_logits(logits, targets, valid, eps=self.eps)
        return self.w_bce * bce + self.w_dice * dice