# src/vesuvius/losses/bce.py
from __future__ import annotations

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """
    Thin wrapper around BCEWithLogitsLoss for consistent call signature.

    Uses logits directly.
    """
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)