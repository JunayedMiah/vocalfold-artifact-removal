"""Downstream task: threshold-based glottis segmentation and GAW extraction.

This mimics the conventional grayscale segmentation the proposal describes:
the glottis is the darkest connected region. Bright laser artifacts break this
assumption, so segmentation quality after preprocessing is the most important
evaluation, more than pixel-level PSNR/SSIM.
"""
from __future__ import annotations

import cv2
import numpy as np


def segment_glottis(img: np.ndarray, rel_dark: float = 0.55, min_area: int = 8) -> np.ndarray:
    """Segment the dark glottis with Otsu thresholding + largest component.

    Args:
        img: float image in [0, 1].
        rel_dark: maximal threshold as a fraction of the median brightness.
        min_area: smaller components are treated as "glottis closed".
    Returns:
        Boolean mask.
    """
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    u8 = cv2.GaussianBlur(u8, (3, 3), 0)
    otsu, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # When the glottis is closed there is no dark region and Otsu would split
    # the tissue instead. Cap the threshold relative to the median brightness,
    # so only clearly dark pixels can become "glottis".
    thr = min(otsu, rel_dark * np.median(u8))
    dark = (u8 < thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return np.zeros(img.shape, bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[biggest, cv2.CC_STAT_AREA] < min_area:
        return np.zeros(img.shape, bool)
    return labels == biggest


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice score; both empty counts as a perfect match (closed glottis)."""
    p, g = pred.astype(bool), gt.astype(bool)
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else 2.0 * (p & g).sum() / s


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    u = (p | g).sum()
    return 1.0 if u == 0 else (p & g).sum() / u


def gaw(masks: np.ndarray) -> np.ndarray:
    """Glottal area waveform: glottis area (pixels) per frame."""
    return masks.reshape(len(masks), -1).sum(1).astype(np.float32)


def gaw_metrics(pred_gaw: np.ndarray, gt_gaw: np.ndarray) -> dict:
    """Compare two GAWs of one sequence."""
    scale = max(gt_gaw.max(), 1.0)
    mae = float(np.mean(np.abs(pred_gaw - gt_gaw)) / scale)   # relative to max area
    if pred_gaw.std() < 1e-6 or gt_gaw.std() < 1e-6:
        r = 0.0
    else:
        r = float(np.corrcoef(pred_gaw, gt_gaw)[0, 1])
    return {"gaw_nmae": mae, "gaw_pearson": r}
