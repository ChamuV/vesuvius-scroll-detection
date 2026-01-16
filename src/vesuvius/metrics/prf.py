# src/vesuvius/metrics/prf.py
from __future__ import annotations
import torch


def prf_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-7,
):
    """
    Precision, Recall, and F1-score for binary 3D segmentation computed from logits.

    The model outputs raw logits which are first converted to probabilities
    via a sigmoid, then thresholded to obtain a binary prediction mask.

    Definitions (per volume, averaged over batch):

    Let:
        P = predicted foreground mask
        G = ground-truth foreground mask

    True positives:
        TP = |P ∩ G|

    False positives:
        FP = |P ∖ G|

    False negatives:
        FN = |G ∖ P|

    Precision:
        precision = TP / (TP + FP)

    Recall:
        recall = TP / (TP + FN)

    F1 score (harmonic mean):
        F1 = 2 * precision * recall / (precision + recall)

    Args:
        logits (torch.Tensor):
            Raw model outputs of shape (B, 1, D, H, W).
        targets (torch.Tensor):
            Binary ground truth masks of shape (B, 1, D, H, W).
        threshold (float, optional):
            Probability threshold used to binarize predictions. Default is 0.5.
        eps (float, optional):
            Small constant added to denominators for numerical stability.

    Returns:
        dict[str, float]:
            {
                "precision": float,
                "recall": float,
                "f1": float,
            }
    """
    # Convert logits → probabilities
    probs = torch.sigmoid(logits)

    # Threshold probabilities → binary prediction mask
    preds = (probs >= threshold).float()

    # Sum over spatial dimensions (D, H, W)
    dims = (2, 3, 4)

    # True positives, false positives, false negatives
    tp = (preds * targets).sum(dims)
    fp = (preds * (1 - targets)).sum(dims)
    fn = ((1 - preds) * targets).sum(dims)

    # Precision, recall, F1 (averaged over batch)
    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()
    f1 = (2 * precision * recall / (precision + recall + eps))

    return {
        "precision": precision,
        "recall": recall,
        "f1": float(f1),
    }