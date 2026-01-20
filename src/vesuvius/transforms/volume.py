# src/vesuvius/transforms/volume.py
from __future__ import annotations
from dataclasses import dataclass
import torch

### MAKE SURE TO LOOK AT MONAI AND TORCHIO
class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample: dict): # Turn instance of call into function
        for transform in self.transforms:
            sample = transform(sample) # apply transform from list one after the other to the obj
        return sample
    

@dataclass
class NormalizeVolume:
    """
    Per-volume normalization or dataset-stat normalization.
    If mean/std provided -> use them. Otherwise -> compute per-sample mean/std.
    Expects image: (1, D, H, W)
    """
    mean: float | None = None
    std: float | None = None
    eps: float | None = 1e-6
    
    def __call__(self, sample:dict):
        img = sample["image"]

        if self.mean is None or self.std is None:
            mean = img.mean()
            std = img.std()
        else:
            mean = torch.as_tensor(self.mean, dtype=img.dtype, device=img.device)
            std = torch.as_tensor(self.std, dtype=img.dtype, device=img.device)
        
        sample["image"] = (img - mean) / (std + self.eps)
        return sample


@dataclass
class CenterCrop3D:
    crop_size: tuple[int, int, int] # (D, H, W)

    def __call__(self, sample: dict):
        img = sample["image"]
        mask = sample.get("mask", None)

        _, D, H, W = img.shape
        cd, ch, cw = self.crop_size 

        if cd > D or ch > H or cw > W:
            raise ValueError(f"Crop {self.crop_size} larger than volume {(D,H,W)}")   

        # Center align
        d0 = (D - cd) // 2
        h0 = (H - ch) // 2
        w0 = (W - cw) // 2

        img = img[:, d0:d0+cd, h0:h0+ch, w0:w0+cw]
        if mask is not None:
            mask = mask[:, d0:d0+cd, h0:h0+ch, w0:w0+cw]

        sample["image"] = img
        if mask is not None:
            sample["mask"] = mask
        return sample


@dataclass
class RandomCrop3D:
    crop_size: tuple[int, int, int]  # (D, H, W)

    def __call__(self, sample: dict):
        img = sample["image"]
        mask = sample.get("mask", None)

        _, D, H, W = img.shape
        cd, ch, cw = self.crop_size

        if cd > D or ch > H or cw > W:
            raise ValueError(f"Crop {self.crop_size} larger than volume {(D,H,W)}")

        d0 = torch.randint(0, D - cd + 1, (1,)).item()
        h0 = torch.randint(0, H - ch + 1, (1,)).item()
        w0 = torch.randint(0, W - cw + 1, (1,)).item()

        img = img[:, d0:d0+cd, h0:h0+ch, w0:w0+cw]
        if mask is not None:
            mask = mask[:, d0:d0+cd, h0:h0+ch, w0:w0+cw]

        sample["image"] = img
        if mask is not None:
            sample["mask"] = mask
        return sample


@dataclass
class RandomFlip3D:
    """
    Randomly flip along selected spatial axes.

    Expects image: (1, D, H, W)
            mask: (1, D, H, W)

    axes can include any of: "D", "H", "W"
    """
    p: float = 0.5
    axes: tuple[str, ...] = ("H", "W")

    def __call__(self, sample: dict):
        img = sample["image"]
        mask = sample.get("mask", None)

        # Map axis name -> tensor dim index
        axis_to_dim = {"D": 1, "H": 2, "W": 3}

        for axis in self.axes:
            if axis not in axis_to_dim:
                raise ValueError(f"Invalid axis '{axis}. Use one of {tuple(axis_to_dim.keys())}.")
            if torch.rand(()) < self.p:
                dim = axis_to_dim[axis]
                img = torch.flip(img, dims=(dim,))
                if mask is not None:
                    mask = torch.flip(mask, dims=(dim,))

        sample["image"] = img
        if mask is not None:
            sample["mask"] = mask
        return sample


@dataclass
class RandomIntensityAffine:
    """
    Apply random affine transform to intensities:
        img <- img * scale + shift

    - scale sampled uniformly from scale_range
    - shift sampled uniformly from shift_range

    Works on image only (mask untouched).
    """
    scale_range: tuple[float, float] = (0.9, 1.1)
    shift_range: tuple[float, float] = (-0.1, 0.1)
    p: float = 0.5

    def __call__(self, sample: dict):
        if torch.rand(()) >= self.p:
            return sample

        img = sample["image"]

        lo_s, hi_s = self.scale_range
        lo_b, hi_b = self.shift_range

        scale = (hi_s - lo_s) * torch.rand((), device=img.device, dtype=img.dtype) + lo_s
        shift = (hi_b - lo_b) * torch.rand((), device=img.device, dtype=img.dtype) + lo_b

        sample["image"] = img * scale + shift
        return sample


@dataclass
class RandomGaussianNoise:
    """
    Additive Gaussian noise:
        img <- img + N(0, sigma^2)

    sigma sampled uniformly from std_range.
    """
    std_range: tuple[float, float] = (0.0, 0.03)
    p: float = 0.5

    def __call__(self, sample: dict):
        if torch.rand(()) >= self.p:
            return sample

        img = sample["image"]
        lo, hi = self.std_range
        sigma = (hi - lo) * torch.rand((), device=img.device, dtype=img.dtype) + lo

        noise = torch.randn_like(img) * sigma
        sample["image"] = img + noise
        return sample