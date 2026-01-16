# src/vesuvius/metrics/dice.py
from __future__ import annotations

import torch


def dice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-7,
):
    """
    Dice score for binary segmentation computed from logits.

    Given predicted binary mask P and ground-truth mask G, the Dice coefficient is

        Dice(P, G) = 2 |P ∩ G| / (|P| + |G|)

    In this implementation:
      - P = 1{ sigmoid(logits) ≥ threshold }
      - G = targets ∈ {0, 1}

    Formally, for each sample:

        Dice = (2 * sum(P ⊙ G) + ε) / (sum(P) + sum(G) + ε)

    where the sum is taken over spatial dimensions (D, H, W), and the final
    value is averaged over the batch.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs of shape (B, 1, D, H, W).
    targets : torch.Tensor
        Ground-truth binary masks of shape (B, 1, D, H, W).
    threshold : float, optional
        Probability threshold applied after sigmoid, by default 0.5.
    eps : float, optional
        Small constant for numerical stability, by default 1e-7.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
          - "dice": mean Dice coefficient over the batch.
    """
    probs = torch.sigmoid(logits) # probs in (0, 1)
    preds = (probs >= threshold).float() # binary mask -> preds in {0, 1}

    dims = (2, 3, 4)
    intersection = (preds * targets).sum(dims)
    denom = preds.sum(dims) + targets.sum(dims)

    dice = ((2 * intersection + eps) / (denom + eps)).mean().item()
    return {"dice": dice}