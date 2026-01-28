# vesuvius/losses/dice_bce.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_dice_loss(logits, targets, eps=1e-6):
    # logits: (B,1,H,W), targets: (B,1,H,W)
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2,3))
    den = (probs + targets).sum(dim=(2,3)).clamp_min(eps)
    dice = num / den
    return 1 - dice.mean()

class DiceBCE(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        dice = soft_dice_loss(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice