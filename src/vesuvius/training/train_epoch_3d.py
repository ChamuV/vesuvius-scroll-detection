# src/vesuvius/training/train_epoch_3d.py
from __future__ import annotations

import inspect
import torch

from vesuvius.losses.targets import make_binary_targets


def train_epoch_3d(model, train_loader, loss_func, optimizer, device) -> float:
    """
    One training epoch for 3D binary segmentation.

    Loader batches: (imgs, masks, sids)
      imgs:  (B, 1, D, H, W)
      masks: (B, 1, D, H, W) with values {0, 1, 2} where 2 = ignore/uncertain

    Model output: logits (B, 1, D, H, W)

    Returns:
        avg_train_loss (float)
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    sig = inspect.signature(loss_func.forward if hasattr(loss_func, "forward") else loss_func)
    accepts_valid = ("valid" in sig.parameters) or (len(sig.parameters) >= 3)

    for imgs, masks, _sids in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        # Convert {0,1,2} -> targets in {0,1}, valid excludes label 2
        targets, valid = make_binary_targets(masks, ignore_label=2)  # targets: float, valid: bool

        optimizer.zero_grad(set_to_none=True)

        logits = model(imgs)

        # Loss: either masked or unmasked depending on what loss_func supports
        if accepts_valid:
            loss = loss_func(logits, targets, valid)
        else:
            loss = loss_func(logits, targets)

        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += float(loss.item()) * bs
        n_samples += bs

    return running_loss / max(1, n_samples)