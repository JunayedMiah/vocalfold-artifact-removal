# Laser-artifact removal for high-speed vocal fold recordings (preparation project)

A small, self-contained project to prepare for the Master's thesis
*"Development of a Neural Network-Based Preprocessing Method for Artifact
Reduction in High-Speed Vocal Fold Recordings"* (Phoniatrics & Pediatric
Audiology, University Hospital Erlangen).

```
clean HSV frame ──► + simulated laser sheet ──► restoration method ──► threshold segmentation ──► GAW
   (+ glottis mask)     (intensity follows the     (3 classical +         (Otsu, darkest region)     (area per frame)
                         glottal opening)           3 CNNs)
```

It answers small versions of the thesis research questions:

| Thesis question | What this project measures |
|---|---|
| RQ1: Can CNNs suppress the artifact and keep the anatomy? | PSNR / SSIM against the clean frame |
| RQ2: Which architecture gives the best trade-off? | 3 CNNs compared on quality, parameters and ms/frame |
| RQ3: Does preprocessing improve segmentation and the GAW? | Dice/IoU of threshold segmentation, GAW error and correlation, Wilcoxon test vs. best classical method |
| RQ4: Generalisation | Artifact parameters are randomised; test sequences are unseen |

## Contents

| File | Purpose |
|---|---|
| `src/artifacts.py` | Laser-sheet artifact model: tilted bright band whose amplitude follows the glottal opening, seeding particles, glare, sensor noise |
| `src/data.py` | Synthetic glottis sequences (no download needed) and a loader for the public BAGLS benchmark |
| `src/models.py` | Baselines (`identity`, `median`, `tophat_inpaint`) and CNNs (`dncnn`, `unet`, `resunet`) |
| `src/segment.py` | Threshold glottis segmentation, Dice/IoU, GAW extraction and GAW metrics |
| `src/train.py` | Training loop (L1 loss, Adam, cosine schedule, best-checkpoint) |
| `src/evaluate.py` | Metrics, runtime, paired Wilcoxon signed-rank test |
| `run.py` | Runs the whole experiment and writes tables and figures to `results/` |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) synthetic data, runs on a laptop CPU
python run.py --data synthetic --epochs 15

# 2) real frames from BAGLS (see below), better on a GPU / the FAU HPC
python run.py --data bagls --bagls-root data/BAGLS --epochs 20 --size 128
```

Outputs in `results/<data>/`:
`summary.csv`, `per_frame_metrics.csv`, `gaw_per_sequence.csv` (synthetic only),
`stats.json`, `examples.png`, `gaw_example.png`, `training_curves.png`.

## Using real frames: BAGLS

BAGLS (Gómez et al., 2020, *Scientific Data* 7:186) is a public benchmark of
59,250 annotated high-speed videoendoscopy frames from 640 recordings in seven
hospitals. Prof. Döllinger is a co-author, so it is the natural public stand-in
for the thesis data.

1. Download from Zenodo: <https://doi.org/10.5281/zenodo.3377544> (licence CC BY-NC-SA 4.0).
2. Unpack so that you have `data/BAGLS/training/0.png, 0_seg.png, …` and `data/BAGLS/test/…`.
3. Start small: `python run.py --data bagls --bagls-root data/BAGLS --limit 3000`.

Notes: BAGLS frames are single frames, so GAW metrics are only computed on the
synthetic sequences. The train/validation split is contiguous to keep frames of
one video together; check the BAGLS metadata for a cleaner split by video.

## First results (synthetic data)

Run: `python run.py --data synthetic --epochs 8 --frames-per-seq 24` on a
2-core CPU (576 training frames, 192 unseen test frames from 8 sequences,
128×128 px). "dice_open" counts only frames where the glottis is open; closed
frames are trivial (empty vs. empty).

| Method | PSNR | SSIM | Dice (open) | GAW nMAE | GAW r | ms/frame | Params |
|---|---|---|---|---|---|---|---|
| clean frame (upper bound) | – | 1.000 | 0.863 | 0.043 | 0.998 | – | – |
| **resunet** | 35.0 | 0.937 | **0.864** | **0.048** | **0.996** | 9.9 | 483k |
| unet | **36.3** | **0.966** | 0.838 | 0.056 | 0.996 | 11.5 | 483k |
| dncnn | 32.4 | 0.884 | 0.742 | 0.115 | 0.949 | 18.7 | 75k |
| tophat_inpaint (best classical) | 27.7 | 0.863 | 0.635 | 0.154 | 0.931 | 2.1 | – |
| identity (no preprocessing) | 22.0 | 0.804 | 0.609 | 0.172 | 0.951 | 0.1 | – |
| median | 23.1 | 0.874 | 0.584 | 0.177 | 0.950 | 0.1 | – |

Wilcoxon signed-rank test on open-frame Dice, ResUNet vs. top-hat + inpainting:
median difference +0.22, p ≈ 2·10⁻¹⁶.

Observations:

* Without preprocessing, the laser sheet cuts the glottis in two and the
  threshold segmentation keeps only one part, so the GAW peaks are roughly halved
  (`gaw_example.png`).
* The residual U-Net recovers almost the full segmentation quality of the clean
  frames. **The best image (highest PSNR/SSIM: U-Net) is not the best for the
  downstream task (ResUNet)**, which supports the proposal's choice of GAW
  extraction as the main evaluation criterion.
* The classical top-hat + inpainting method removes particles but not the broad
  sheet where it crosses the dark glottis.

**Limitations (be honest about these in the meeting):** the data and artifacts
are simulated, the synthetic glottis is much simpler than real endoscopy, there is
one training seed only, and the training was short. These numbers show that the
pipeline works, not how well the method will work on the real recordings.

## Next steps

| Step | Task | Why |
|---|---|---|
| 1 | Run on BAGLS frames (start with `--limit 3000`) | Real anatomy and texture |
| 2 | 3 seeds per model, mean ± std | Thesis-level reporting |
| 3 | Add an L1 + SSIM or segmentation-aware loss | Optimise for the downstream task, not only pixels |
| 4 | Add one modern restoration model (e.g. NAFNet-style block) | Thesis asks for 3–5 architectures |
| 5 | Vary artifact strength and test outside the training range | RQ4: generalisation |
| 6 | Read Kist et al. 2021 (GAT software) and the BAGLS paper | Talk the group's language in the meeting |

## References

* Gómez P. et al. (2020). BAGLS, a multihospital Benchmark for Automatic Glottis Segmentation. *Scientific Data* 7:186. doi:10.1038/s41597-020-0526-3
* Kist A. M. et al. (2021). A Deep Learning Enhanced Novel Software Tool for Laryngeal Dynamics Analysis.
* Ronneberger O. et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
* Zhang K. et al. (2017). Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (DnCNN). IEEE TIP.
