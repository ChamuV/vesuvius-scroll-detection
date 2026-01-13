# src/vesuvius/transforms/volume.py
from __future__ import annotations
from dataclasses import dataclass
import torch

### MAKE SURE TO LOOK AT MONAI AND TORCHIO
class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, sample: dict): # Turn instance of call into function
        for t in self.transforms:
            sample = t(sample) # apply transform from list one after the other to the obj
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
