# vesuvius/transforms/volume25d.py
import torch
import random

class RandomCrop25D:
    """
    Takes a 3D volume (D,H,W) and 2D mask (H,W) or (1,H,W),
    returns:
      x: (C,Hc,Wc) where C=num_slices (2.5D)
      y: (1,Hc,Wc)
    """
    def __init__(self, crop_hw=(256, 256), num_slices=32):
        self.crop_h, self.crop_w = crop_hw
        self.num_slices = num_slices

    def __call__(self, img: torch.Tensor, mask: torch.Tensor):
        # img expected (D,H,W) or (1,D,H,W)
        if img.dim() == 4:
            img = img.squeeze(0)  # -> (D,H,W)

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # -> (1,H,W)

        D, H, W = img.shape
        ch, cw = self.crop_h, self.crop_w

        if H < ch or W < cw:
            raise ValueError(f"Crop {ch}x{cw} bigger than volume {H}x{W}")

        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)

        # pick a center slice safely
        half = self.num_slices // 2
        z_center = random.randint(half, D - half - 1)

        z0 = z_center - half
        z1 = z0 + self.num_slices

        x = img[z0:z1, top:top+ch, left:left+cw]          # (C,H,W)
        y = mask[:, top:top+ch, left:left+cw]             # (1,H,W)

        return x, y


class NormalizePerPatch:
    """Simple robust normalization per patch."""
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        # x: (C,H,W)
        mean = x.mean()
        std = x.std().clamp_min(self.eps)
        x = (x - mean) / std
        return x, y