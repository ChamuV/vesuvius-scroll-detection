# src/vesuvius/training/validate_epoch_3d.py
from __future__ import annotations

import torch

from vesuvius.losses.targets import make_binary_targets
from vesuvius.metrics import dice_from_logits


def validate_epoch_3d(
    model,
    val_loader,
    loss_fn,
    device,
    *,
    metric_fn=dice_from_logits,
    threshold: float = 0.5,
):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    metric_sums: dict[str, float] = {}

    with torch.no_grad():
        for imgs, masks, _sids in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            targets, valid = make_binary_targets(masks, ignore_label=2)

            logits = model(imgs)
            loss = loss_fn(logits, targets, valid)

            bs = imgs.size(0)
            running_loss += float(loss.item()) * bs
            n_samples += bs

            batch_metrics = metric_fn(logits, targets, threshold=threshold)
            for k, v in batch_metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v) * bs

    avg_val_loss = running_loss / max(1, n_samples)
    avg_metrics = {k: v / max(1, n_samples) for k, v in metric_sums.items()}
    return avg_val_loss, avg_metrics