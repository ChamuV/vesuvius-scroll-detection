# src/vesuvius/losses/bce.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    """
    BCEWithLogits with optional mask (valid).
    If valid is provided, ignored voxels do not contribute to the loss.
    """
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid is None:
            return F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=self.pos_weight,
            )

        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        v = valid.float()
        loss = loss * v
        denom = v.sum().clamp_min(1.0)
        return loss.sum() / denom