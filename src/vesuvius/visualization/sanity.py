# src/vesuvius/visualization/sanity.py
from __future__ import annotations

import torch
import matplotlib.pyplot as plt


def sanity_visualization(
    model,
    loader,
    device,
    *,
    threshold: float = 0.3,
    slice_index: int | None = None,
    figsize=(18, 6),
):
    """
    Visual sanity check for 3D segmentation.

    Shows:
      1) GT overlay
      2) Predicted probability overlay
      3) Thresholded prediction overlay

    Assumes loader yields (imgs, masks, sids) where:
      imgs  : (B,1,D,H,W)
      masks : (B,1,D,H,W) with labels {0,1,2}
    """
    model.eval()

    imgs, masks, sids = next(iter(loader))
    imgs = imgs.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu()

    img = imgs.cpu()[0]      # (1,D,H,W)
    mask = masks.cpu()[0]    # (1,D,H,W)
    prob = probs[0]          # (1,D,H,W)
    sid = sids[0]

    D = img.shape[1]
    z = slice_index if slice_index is not None else D // 2

    plt.figure(figsize=figsize)

    # 1) GT overlay
    plt.subplot(1, 3, 1)
    plt.imshow(img[0, z], cmap="gray")
    plt.imshow((mask[0, z] == 1), alpha=0.35, cmap="Reds")
    plt.title(f"{sid} | GT overlay | z={z}")
    plt.axis("off")

    # 2) Pred prob overlay
    plt.subplot(1, 3, 2)
    plt.imshow(img[0, z], cmap="gray")
    plt.imshow(prob[0, z], alpha=0.35, cmap="Reds", vmin=0.0, vmax=1.0)
    plt.title(f"{sid} | Pred prob overlay | z={z}")
    plt.axis("off")

    # 3) Pred mask overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img[0, z], cmap="gray")
    plt.imshow((prob[0, z] > threshold), alpha=0.35, cmap="Reds")
    plt.title(f"{sid} | Pred mask (thr={threshold}) | z={z}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()