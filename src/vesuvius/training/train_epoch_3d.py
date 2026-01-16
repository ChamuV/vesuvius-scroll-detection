# src/vesuvius/training/train_epoch_3d.py
from __future__ import annotations

import torch

def train_epoch_3d(model, train_loader, loss_func, optimizer, device) -> float:
    """
    One training epoch for 3D binary segmentation.

    Expects loader batches: (imgs, masks, sids)
      imgs:  (B, 1, D, H, W)
      masks: (B, 1, D, H, W)  (float or int; we'll cast to float)
    Model output: logits (B, 1, D, H, W)

    Returns:
        avg_train_loss (float)
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    for imgs, masks, _sids in train_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs) 
        loss = loss_func(outputs, masks)
        loss.backwards()
        optimizer.step()

        bs = imgs.size(0) # batch size
        running_loss += loss.item() * bs 
        n_samples += bs
    
    return running_loss / max(1, n_samples) # average loss