from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tifffile as tiff


def _as_numpy(path_or_arr):
    if path_or_arr is None:
        return None
    if isinstance(path_or_arr, (str, Path)):
        return tiff.imread(str(path_or_arr))
    return np.asarray(path_or_arr)


def _percentile_limits(vol: np.ndarray, lo=1.0, hi=99.0) -> Tuple[float, float]:
    v = vol.astype(np.float32)
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def _make_surface_from_mask(mask: np.ndarray):
    """
    Convert a 3D binary mask into a triangle mesh using marching cubes.
    Returns (verts, faces) in napari format.
    """
    from skimage.measure import marching_cubes  # lazy import

    m = (mask > 0).astype(np.uint8)
    if m.ndim != 3:
        raise ValueError(f"Surface rendering expects a 3D mask. Got shape={m.shape}")

    if m.sum() == 0:
        return None

    # marching cubes expects values; level=0.5 extracts boundary between 0 and 1
    verts, faces, _, _ = marching_cubes(m, level=0.5)
    return verts, faces


def view_volume_with_mask(
    img: str | Path | np.ndarray,
    mask: Optional[str | Path | np.ndarray] = None,
    *,
    ndisplay: int = 3,
    mask_mode: str = "surface",  # "surface" (best) OR "volume"
    mask_opacity: float = 0.35,
    image_rendering: str = "mip",  # "mip", "attenuated_mip", "translucent"
    image_clim_percentiles: Tuple[float, float] = (1.0, 99.0),
):
    """
    Open a CT volume (and optional mask) in napari.

    Parameters
    ----------
    img:
        Path or array. Can be 2D or 3D.
    mask:
        Optional Path or array. Often 2D or 3D binary/label mask.
    ndisplay:
        2 or 3. For 3D volumes use 3.
    mask_mode:
        "surface": render mask as a mesh (best for thin sheets).
        "volume": render mask as translucent volume.
    """
    import napari

    img_arr = _as_numpy(img)
    mask_arr = _as_numpy(mask)

    viewer = napari.Viewer(ndisplay=ndisplay)

    # Image layer
    clim = _percentile_limits(img_arr, *image_clim_percentiles)
    viewer.add_image(
        img_arr,
        name="img",
        contrast_limits=clim,
        rendering=image_rendering if (ndisplay == 3 and img_arr.ndim == 3) else "nearest",
        blending="translucent_no_depth",
    )

    # Mask layer 
    if mask_arr is not None:
        # If mask is 2D but image is 3D, that's fine (it will show as a single slice overlay),
        # but surface mode requires 3D.
        if mask_mode == "surface":
            if mask_arr.ndim == 3:
                surf = _make_surface_from_mask(mask_arr)
                if surf is not None:
                    verts, faces = surf
                    viewer.add_surface(
                        (verts, faces),
                        name="mask_surface",
                        opacity=0.85,
                    )
                else:
                    print("Mask is empty; no surface to render.")
            else:
                # fallback: show as labels in 2D
                viewer.add_labels(mask_arr, name="mask_labels")
        elif mask_mode == "volume":
            # show mask voxels only (binary), translucent, in red colormap
            m = (mask_arr > 0).astype(np.float32)
            viewer.add_image(
                m,
                name="mask_volume",
                colormap="red",
                opacity=mask_opacity,
                rendering="translucent" if (ndisplay == 3 and m.ndim == 3) else "nearest",
                blending="additive",
                contrast_limits=(0.0, 1.0),
            )
        else:
            raise ValueError("mask_mode must be 'surface' or 'volume'")

    napari.run()