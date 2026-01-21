# src/vesuvius/training/train_epoch_3d.py
from __future__ import annotations

import torch
from vesuvius.losses.targets import make_binary_targets


def train_epoch_3d(model, train_loader, loss_fn, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for imgs, masks, _sids in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        targets, valid = make_binary_targets(masks, ignore_label=2)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)

        loss = loss_fn(logits, targets, valid)  # always same call signature

        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += float(loss.item()) * bs
        n_samples += bs

    return running_loss / max(1, n_samples)