"""Data sources.

Two sources are supported:

* ``synthetic``: procedurally generated glottis sequences. Runs anywhere with
  no download, so the whole pipeline can be tested on a laptop in minutes.
* ``bagls``: real high-speed videoendoscopy frames from the public BAGLS
  benchmark (Gómez et al., 2020, Scientific Data), which comes from the same
  research environment as the thesis (Prof. Döllinger is a co-author).
  Expected layout (as in the BAGLS download)::

      <root>/training/0.png, 0_seg.png, 1.png, 1_seg.png, ...
      <root>/test/0.png, 0_seg.png, ...

Both return clean frames, glottis masks and an "opening" value per frame that
drives the artifact intensity.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .artifacts import add_laser_artifact, sample_params


# ---------------------------------------------------------------- synthetic
def synthetic_sequence(
    rng: np.random.Generator, n_frames: int = 40, size: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one synthetic high-speed sequence.

    The glottis is a dark spindle between two bright vocal folds; its width
    follows a periodic, asymmetric oscillation (fast closing, slower opening).

    Returns:
        frames (T, H, W) in [0, 1], masks (T, H, W) bool, opening (T,) in [0, 1].
    """
    h = w = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Static tissue texture (low-frequency noise) and vignette.
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 6)
    tex = (tex - tex.min()) / (np.ptp(tex) + 1e-8)
    vign = 1 - 0.5 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    tissue = np.clip(0.45 + 0.25 * tex, 0, 1) * np.clip(vign, 0.6, 1)

    cx = w / 2 + rng.uniform(-8, 8)
    top, bottom = h * rng.uniform(0.15, 0.25), h * rng.uniform(0.75, 0.85)
    tilt = rng.uniform(-0.15, 0.15)
    max_half_width = rng.uniform(6, 14)
    period = rng.uniform(8, 14)
    phase = rng.uniform(0, 2 * np.pi)

    frames, masks, opening = [], [], []
    for t in range(n_frames):
        s = np.sin(2 * np.pi * t / period + phase)
        o = max(0.0, s) ** 0.8                     # closed for half the cycle
        # Spindle shape: widest in the middle, zero at both ends.
        rel = np.clip((yy - top) / (bottom - top), 0, 1)
        half = o * max_half_width * np.clip(np.sin(np.pi * rel), 0, None) ** 0.7
        centre = cx + tilt * (yy - h / 2)
        inside = (np.abs(xx - centre) < half) & (yy > top) & (yy < bottom)

        img = tissue.copy()
        img[inside] = 0.05 + 0.03 * rng.random(inside.sum())   # dark glottis
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        frames.append(img.astype(np.float32))
        masks.append(inside)
        opening.append(o)
    return np.stack(frames), np.stack(masks), np.asarray(opening, np.float32)


# ---------------------------------------------------------------- BAGLS
def load_bagls_folder(folder: str | Path, size: int = 128, limit: int | None = None):
    """Load BAGLS frames and masks from one folder, resized to ``size``.

    Opening is estimated per frame as mask area / largest area in the folder,
    so the artifact still follows the glottal opening.
    """
    folder = Path(folder)
    ids = sorted(
        int(p.stem) for p in folder.glob("*.png") if not p.stem.endswith("_seg")
    )
    if limit:
        ids = ids[:limit]
    frames, masks = [], []
    for i in ids:
        img = cv2.imread(str(folder / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        seg = cv2.imread(str(folder / f"{i}_seg.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or seg is None:
            continue
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        seg = cv2.resize(seg, (size, size), interpolation=cv2.INTER_NEAREST)
        frames.append(img.astype(np.float32) / 255.0)
        masks.append(seg > 127)
    frames, masks = np.stack(frames), np.stack(masks)
    area = masks.reshape(len(masks), -1).sum(1).astype(np.float32)
    opening = area / max(area.max(), 1.0)
    return frames, masks, opening


# ---------------------------------------------------------------- dataset
class ArtifactPairs(Dataset):
    """(artifact, clean, mask) triplets with artifacts simulated on the fly.

    Simulating on the fly means every epoch sees new artifact variations,
    which acts as strong data augmentation.
    """

    def __init__(self, frames, masks, opening, seed: int = 0, fixed: bool = False):
        self.frames, self.masks, self.opening = frames, masks, opening
        self.seed, self.fixed = seed, fixed
        self.epoch = 0

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # Fixed mode (validation/test): same artifact for an index every time.
        s = self.seed * 100003 + idx if self.fixed else None
        rng = np.random.default_rng(s)
        params = sample_params(rng)
        clean = self.frames[idx]
        noisy = add_laser_artifact(clean, float(self.opening[idx]), params, rng)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None]
        return to_t(noisy), to_t(clean), to_t(self.masks[idx].astype(np.float32))


def build_synthetic(n_seq: int, n_frames: int, size: int, seed: int):
    """Concatenate several synthetic sequences into frame arrays."""
    rng = np.random.default_rng(seed)
    f, m, o, seq_id = [], [], [], []
    for k in range(n_seq):
        fr, ma, op = synthetic_sequence(rng, n_frames, size)
        f.append(fr); m.append(ma); o.append(op); seq_id += [k] * n_frames
    return np.concatenate(f), np.concatenate(m), np.concatenate(o), np.asarray(seq_id)
