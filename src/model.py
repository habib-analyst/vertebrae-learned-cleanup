"""Tiny 3D U-Net cleanup network for Project #7.

Input:  (intensity[1] + one-hot baseline mask[9]) = 10 channels, 64^3 patch.
Output: 9-class logits (background + 8 vertebrae).

Kept deliberately small (~60k params) so it trains on CPU in reasonable time
and would fit any free-tier GPU for the VerSe-scale follow-up.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet3D(nn.Module):
    def __init__(self, in_ch=10, n_classes=9, base=8):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.head = nn.Conv3d(base, n_classes, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        y = self.up2(x3)
        y = self.d2(torch.cat([y, x2], 1))
        y = self.up1(y)
        y = self.d1(torch.cat([y, x1], 1))
        return self.head(y)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
