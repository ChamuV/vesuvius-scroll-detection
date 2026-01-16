# src/vesuvius/losses/masked_bce.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # logits/targets/valid: (B,1,D,H,W)
        loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        loss = loss * valid.float()
        denom = valid.float().sum().clamp_min(1.0)
        return loss.sum() / denom