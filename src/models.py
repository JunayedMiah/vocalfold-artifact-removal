"""Restoration models: classical baselines and three small CNNs.

Classical baselines (no training):
    * ``identity``       – do nothing (lower bound).
    * ``median``         – median filter.
    * ``tophat_inpaint`` – detect bright structures with a morphological white
                           top-hat, then inpaint them (closest to the
                           "conventional preprocessing" in the proposal).

Deep models (trained, all predict the clean image):
    * ``dncnn``    – plain residual CNN (Zhang et al., 2017), no down-sampling.
    * ``unet``     – encoder–decoder with skip connections (Ronneberger 2015).
    * ``resunet``  – U-Net that predicts the *artifact* and subtracts it
                     (residual learning), usually easier to train.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn


# ================================================================ classical
def classical_restore(img: np.ndarray, method: str) -> np.ndarray:
    """Apply a non-learned baseline to one float image in [0, 1]."""
    if method == "identity":
        return img
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if method == "median":
        return cv2.medianBlur(u8, 5).astype(np.float32) / 255.0
    if method == "tophat_inpaint":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, kernel)
        _, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        out = cv2.inpaint(u8, mask, 5, cv2.INPAINT_TELEA)
        return out.astype(np.float32) / 255.0
    raise ValueError(f"unknown classical method: {method}")


CLASSICAL = ["identity", "median", "tophat_inpaint"]


# ================================================================ deep
class DnCNN(nn.Module):
    """Plain conv stack that predicts the residual (artifact)."""

    def __init__(self, depth: int = 10, ch: int = 32):
        super().__init__()
        layers = [nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(ch, 1, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(x - self.net(x), 0, 1)


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Small 3-level U-Net. ``residual=True`` gives the ResUNet variant."""

    def __init__(self, base: int = 16, residual: bool = False):
        super().__init__()
        self.residual = residual
        b = base
        self.e1, self.e2, self.e3 = _block(1, b), _block(b, 2 * b), _block(2 * b, 4 * b)
        self.bott = _block(4 * b, 8 * b)
        self.pool = nn.MaxPool2d(2)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), _block(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), _block(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), _block(2 * b, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        z = self.bott(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(z), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        y = self.out(d1)
        if self.residual:
            return torch.clamp(x - y, 0, 1)      # predict the artifact
        return torch.sigmoid(y)                  # predict the clean image


def build_model(name: str) -> nn.Module:
    if name == "dncnn":
        return DnCNN()
    if name == "unet":
        return UNet(residual=False)
    if name == "resunet":
        return UNet(residual=True)
    raise ValueError(f"unknown model: {name}")


DEEP = ["dncnn", "unet", "resunet"]


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
