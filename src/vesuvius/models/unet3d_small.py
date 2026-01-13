# src/vesuvius/models/unet3d_small.py
from __future__ import annotations
import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv = DoubleConv3D(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        # after upsample, we concat with skip => out_ch + out_ch = 2*out_ch
        self.conv = DoubleConv3D(in_ch=out_ch * 2, out_ch=out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # handle odd sizes by center-cropping skip if needed
        if x.shape[-3:] != skip.shape[-3:]:
            dz = (skip.shape[-3] - x.shape[-3]) // 2
            dy = (skip.shape[-2] - x.shape[-2]) // 2
            dx = (skip.shape[-1] - x.shape[-1]) // 2
            skip = skip[
                :,
                :,
                dz : dz + x.shape[-3],
                dy : dy + x.shape[-2],
                dx : dx + x.shape[-1],
            ]

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SmallUNet3D(nn.Module):
    """
    Input:  (B, 1, D, H, W)
    Output: (B, 1, D, H, W) logits
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        c = base_channels

        self.inc = DoubleConv3D(in_channels, c)
        self.down1 = Down3D(c, c * 2)
        self.down2 = Down3D(c * 2, c * 4)

        self.up1 = Up3D(c * 4, c * 2)
        self.up2 = Up3D(c * 2, c)

        self.outc = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)