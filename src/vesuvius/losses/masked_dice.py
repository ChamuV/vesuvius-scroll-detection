# src/vesuvius/losses/masked_dice.py
import torch

def masked_dice_loss_from_logits(logits, targets, valid, eps=1e-6):
    probs = torch.sigmoid(logits) * valid.float()
    t = targets * valid.float()
    dims = (2,3,4)
    inter = (probs * t).sum(dims)
    denom = probs.sum(dims) + t.sum(dims)
    dice = (2*inter + eps) / (denom + eps)
    return 1.0 - dice.mean()