# src/vesuvius/visualization/patch_grid.py
from __future__ import annotations

import torch
import matplotlib.pyplot as plt
from typing import Iterable, Optional


def _to_numpy_2d(x: torch.Tensor) -> torch.Tensor:
    """
    x: (1, D, H, W) or (1, H, W)
    returns: (H, W) tensor on cpu
    """
    x = x.detach().float().cpu()
    if x.ndim == 4:  # (1, D, H, W)
        z = x.shape[1] // 2
        return x[0, z]
    elif x.ndim == 3:  # (1, H, W)
        return x[0]
    else:
        raise ValueError(f"Unexpected tensor shape: {tuple(x.shape)}")


def plot_transformed_patch_grid(
    dataset,
    indices: Optional[Iterable[int]] = None,
    *,
    n: int = 6,
    seed: int = 0,
    title: str = "Transformed patches",
    cmap_img: str = "gray",
    cmap_mask: str = "Reds",
    alpha: float = 0.35,
):
    """
    Visualize transformed samples from a dataset that returns (img, mask, sid),
    where img/mask are torch tensors after transforms.

    dataset: e.g. train_ds (SubsetWithTransform)
    indices: explicit indices to show. If None, randomly sample n indices.
    n: number of samples if indices is None.
    """
    # Choose indices
    if indices is None:
        g = torch.Generator().manual_seed(seed)
        idxs = torch.randperm(len(dataset), generator=g)[:n].tolist()
    else:
        idxs = list(indices)
        n = len(idxs)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]  # Index consistency

    for j, idx in enumerate(idxs):
        img, mask, sid = dataset[idx]

        img2d = _to_numpy_2d(img)
        mask2d = _to_numpy_2d(mask)

        # Normalise images to [0, 1]
        mn, mx = img2d.min(), img2d.max()
        img_disp = (img2d - mn) / (mx - mn + 1e-8)

        axes[0][j].imshow(img_disp, cmap=cmap_img)
        axes[0][j].set_title(f"{sid}\nidx={idx}")
        axes[0][j].axis("off")

        axes[1][j].imshow(img_disp, cmap=cmap_img)
        axes[1][j].imshow((mask2d > 0), alpha=alpha, cmap=cmap_mask)
        axes[1][j].set_title("overlay")
        axes[1][j].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()