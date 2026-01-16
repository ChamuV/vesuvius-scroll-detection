from __future__ import annotations
import torch

def make_binary_targets(masks: torch.Tensor, *, ignore_label: int = 2):
    """
    masks: (B,1,D,H,W) with values {0,1,2}
    returns:
      targets: float {0,1} where 1 means label==1
      valid  : bool mask where label != ignore_label
    """
    valid = (masks != ignore_label)
    targets = (masks == 1).float()
    return targets, valid