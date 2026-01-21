# src/vesuvius/visualization/prediction_viz.py
from __future__ import annotations

import torch
import matplotlib.pyplot as plt


def plot_prediction_triplet(
    *,
    model,
    loader,
    device,
    threshold: float = 0.3,
    slice_index: int | None = None,
    title_prefix: str = "",
):
    model.eval()
    imgs, masks, sids = next(iter(loader))
    imgs = imgs.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(imgs)).detach().cpu()

    img = imgs.detach().cpu()[0]     # (1,D,H,W)
    mask = masks.detach().cpu()[0]   # (1,D,H,W)
    prob = probs[0]                  # (1,D,H,W)
    sid = sids[0]

    D = img.shape[1]
    z = slice_index if slice_index is not None else D // 2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # GT overlay
    axes[0].imshow(img[0, z], cmap="gray")
    axes[0].imshow((mask[0, z] == 1), alpha=0.35, cmap="Reds")
    axes[0].set_title(f"{title_prefix} | GT | {sid}")
    axes[0].axis("off")

    # Pred prob overlay + colorbar
    axes[1].imshow(img[0, z], cmap="gray")
    im = axes[1].imshow(prob[0, z], cmap="magma", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[1].set_title(f"{title_prefix} | Pred prob (0..1) | z={z}")
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("P(surface)")

    # Pred mask overlay
    axes[2].imshow(img[0, z], cmap="gray")
    axes[2].imshow((prob[0, z] > threshold), alpha=0.35, cmap="Reds")
    axes[2].set_title(f"{title_prefix} | Pred mask (thr={threshold}) | z={z}")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()