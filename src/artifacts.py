"""Simulation of laser-sheet illumination artifacts on high-speed vocal fold frames.

The thesis data contain a bright laser sheet (used to visualise airflow/particles
above the glottis). Its intensity changes during the oscillation cycle as the
glottis opens and closes, so a fixed threshold fails. We do not have such data
yet, so we *simulate* the artifact on clean frames. This gives paired
(artifact, clean) images for supervised training and a ground-truth mask for
downstream evaluation.

Artifact model (all parameters randomised per sequence, varied per frame):
    1. Laser sheet: a bright, slightly tilted band with Gaussian cross-profile.
    2. Intensity modulation: amplitude = a0 + a1 * opening(t), i.e. it follows
       the glottal opening, as described in the thesis proposal.
    3. Particles: small bright specks scattered along the sheet (seeding).
    4. Glare: a soft specular blob near the sheet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArtifactParams:
    angle_deg: float      # tilt of the laser sheet
    center: float         # sheet position as a fraction of image height (0..1)
    width: float          # sheet half-width in pixels (Gaussian sigma)
    base_amp: float       # amplitude when the glottis is closed
    open_gain: float      # extra amplitude proportional to the opening
    n_particles: int      # number of seeding particles per frame
    glare_amp: float      # amplitude of the specular glare blob
    glare_xy: tuple       # glare centre as fractions (x, y)


def sample_params(rng: np.random.Generator) -> ArtifactParams:
    """Draw random artifact parameters for one sequence."""
    return ArtifactParams(
        angle_deg=float(rng.uniform(-25, 25)),
        center=float(rng.uniform(0.3, 0.7)),
        width=float(rng.uniform(2.0, 7.0)),
        base_amp=float(rng.uniform(0.15, 0.35)),
        open_gain=float(rng.uniform(0.25, 0.55)),
        n_particles=int(rng.integers(10, 60)),
        glare_amp=float(rng.uniform(0.0, 0.35)),
        glare_xy=(float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))),
    )


def add_laser_artifact(
    clean: np.ndarray,
    opening: float,
    params: ArtifactParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add a laser-sheet artifact to one frame.

    Args:
        clean: float image in [0, 1], shape (H, W).
        opening: normalised glottal opening of this frame in [0, 1]
            (0 = closed, 1 = maximally open in the sequence).
        params: sequence-level artifact parameters.
        rng: random generator for per-frame jitter.

    Returns:
        Image with artifact, clipped to [0, 1].
    """
    h, w = clean.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # 1) Laser sheet: distance of each pixel to a tilted line.
    theta = np.deg2rad(params.angle_deg)
    cy = params.center * h + rng.normal(0, 1.0)          # small per-frame jitter
    cx = w / 2.0
    dist = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
    sheet = np.exp(-0.5 * (dist / params.width) ** 2)

    # 2) Intensity follows the glottal opening (the core difficulty).
    amp = params.base_amp + params.open_gain * opening
    amp *= rng.uniform(0.9, 1.1)
    artifact = amp * sheet

    # 3) Seeding particles along the sheet.
    for _ in range(params.n_particles):
        t = rng.uniform(-0.5, 0.5) * w
        px = cx + t * np.cos(theta)
        py = cy + t * np.sin(theta) + rng.normal(0, params.width)
        r = rng.uniform(0.7, 1.8)
        artifact += rng.uniform(0.3, 0.8) * np.exp(
            -((xx - px) ** 2 + (yy - py) ** 2) / (2 * r ** 2)
        )

    # 4) Specular glare blob.
    if params.glare_amp > 0:
        gx, gy = params.glare_xy[0] * w, params.glare_xy[1] * h
        artifact += params.glare_amp * (0.5 + 0.5 * opening) * np.exp(
            -((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * (0.08 * w) ** 2)
        )

    # Screen-like blending: bright light lifts dark regions most.
    out = 1.0 - (1.0 - clean) * (1.0 - np.clip(artifact, 0, 1))
    out += rng.normal(0, 0.01, size=out.shape)            # sensor noise
    return np.clip(out, 0.0, 1.0).astype(np.float32)
