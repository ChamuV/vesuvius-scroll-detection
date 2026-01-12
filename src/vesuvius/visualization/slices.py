from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional

import tifffile as tiff

def get_slice(
        vol: np.array,
        axis: int = 0,
        z: int | None = None,
) -> np.array:
    """
    Extract a 2D slice from a 3D volume.

    Parameters
    ----------
    vol : np.ndarray
        2D or 3D array.
    axis : int
        Axis along which to slice (only used if vol is 3D).
    z : int or None
        Slice index. If None, use middle slice.

    Returns
    -------
    np.ndarray
        2D slice.
    """
    if vol.ndim == 2:
        return vol
    if vol.ndim != 3:
        raise ValueError(f"Expected 2D or 3D array, got shape {vol.shape}")
    
    if z is None:
        z = vol.shape[axis] // 2 # middle slice
    
    return np.take(vol, z, axis=axis)

def normalize01(x: np.ndarray) -> np.ndarray:
    """
    Normalize array to [0, 1] for visualization.
    """
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)

def plot_slice_grid(
        ids: Iterable[str],
        img_dir,
        mask_dir,
        *,
        axis: int = 0,
        z: Optional[int] = None,
        n_cols: int = 6,
        cmap_img: str = "gray",
        cmap_mask: str = "Reds",
):
    """
    Plot a grid of image slices with mask overlays.

    Parameters
    ----------
    ids : iterable of str
        Sample IDs to plot.
    img_dir, mask_dir : Path
        Directories containing .tif volumes.
    axis : int
        Slice axis.
    z : int or None
        Slice index (None = middle slice).
    n_cols : int
        Number of columns.
    """
    ids = list(ids)
    fig, axes = plt.subplots(2, len(ids), figsize=(3 * len(ids), 6))

    for j, sid in enumerate(ids):
        vol = tiff.imread(img_dir / f"{sid}.tif")
        mvol = tiff.imread(mask_dir / f"{sid}.tif")

        img2d = normalize01(get_slice(vol, axis=axis, z=z))
        mask2d = get_slice(mvol, axis=axis, z=z)

        z_eff = vol.shape[axis] // 2 if z is None else z

        axes[0, j].imshow(img2d, cmap=cmap_img)
        axes[0, j].set_title(f"img\n{sid}\naxis={axis}, z={z_eff}")
        axes[0, j].axis("off")

        axes[1, j].imshow(img2d, cmap=cmap_img)
        axes[1, j].imshow(mask2d > 0, alpha=0.35, cmap=cmap_mask)
        axes[1, j].set_title("overlay")
        axes[1, j].axis("off")

    plt.tight_layout()
    plt.show()