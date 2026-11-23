"""End-to-end experiment: simulate artifacts, train models, evaluate, plot.

Examples:
    # quick test on synthetic data (no download needed, ~10 min on a laptop CPU)
    python run.py --data synthetic --epochs 15

    # real frames from the BAGLS benchmark
    python run.py --data bagls --bagls-root /path/to/BAGLS --epochs 20 --size 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import ArtifactPairs, build_synthetic, load_bagls_folder
from src.evaluate import gaw_table, paired_test, per_frame_metrics, restore_all, summarise
from src.models import CLASSICAL, DEEP
from src.segment import gaw, segment_glottis
from src.train import train_model


def get_data(args):
    """Return train/val/test dicts with frames, masks, opening, seq_id."""
    if args.data == "synthetic":
        def mk(n_seq, seed):
            f, m, o, s = build_synthetic(n_seq, args.frames_per_seq, args.size, seed)
            return dict(frames=f, masks=m, opening=o, seq_id=s)
        # Different seeds -> different sequences: no leakage between splits.
        return mk(24, 1), mk(4, 2), mk(8, 3)

    root = Path(args.bagls_root)
    f, m, o = load_bagls_folder(root / "training", args.size, args.limit)
    cut = int(0.9 * len(f))                     # contiguous split (frames of a video are adjacent)
    train = dict(frames=f[:cut], masks=m[:cut], opening=o[:cut], seq_id=np.zeros(cut, int))
    val = dict(frames=f[cut:], masks=m[cut:], opening=o[cut:], seq_id=np.zeros(len(f) - cut, int))
    tf, tm, to = load_bagls_folder(root / "test", args.size, args.limit)
    test = dict(frames=tf, masks=tm, opening=to, seq_id=np.arange(len(tf)))  # single frames: no GAW
    return train, val, test


def plot_examples(out, noisy, clean, masks, restored, n=4):
    methods = list(restored)
    idx = np.argsort(-masks.reshape(len(masks), -1).sum(1))[:: max(1, len(masks) // n)][:n]
    cols = ["clean", "with artifact"] + methods
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(2.1 * len(cols), 2.2 * len(idx)))
    for r, i in enumerate(idx):
        imgs = [clean[i], noisy[i]] + [restored[m][i] for m in methods]
        for c, (title, im) in enumerate(zip(cols, imgs)):
            ax = axes[r, c]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            seg = segment_glottis(im)
            ax.contour(masks[i], levels=[0.5], colors="lime", linewidths=0.8)
            ax.contour(seg, levels=[0.5], colors="red", linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
    fig.suptitle("green = ground-truth glottis, red = threshold segmentation", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=130)
    plt.close(fig)


def plot_gaw(out, test, restored, methods):
    s = test["seq_id"]
    idx = np.where(s == s[0])[0]
    t = np.arange(len(idx))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(t, gaw(test["masks"][idx]), "k-", lw=2.5, label="ground truth")
    for m in methods:
        pred = np.stack([segment_glottis(x) for x in restored[m][idx]])
        ax.plot(t, gaw(pred), lw=1.3, label=m)
    ax.set_xlabel("frame"); ax.set_ylabel("glottal area [px]")
    ax.set_title("Glottal area waveform (one test sequence)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out / "gaw_example.png", dpi=130)
    plt.close(fig)


def plot_curves(out, histories):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for name, h in histories.items():
        h = pd.DataFrame(h)
        ax.plot(h.epoch, h.val_l1, label=f"{name} (val)")
        ax.plot(h.epoch, h.train_l1, "--", alpha=0.5, label=f"{name} (train)")
    ax.set_xlabel("epoch"); ax.set_ylabel("L1 loss"); ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "bagls"], default="synthetic")
    ap.add_argument("--bagls-root", default=None)
    ap.add_argument("--limit", type=int, default=None, help="max frames per BAGLS folder")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--frames-per-seq", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--models", nargs="+", default=DEEP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out) / args.data
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}")

    train, val, test = get_data(args)
    print(f"frames  train {len(train['frames'])}  val {len(val['frames'])}  test {len(test['frames'])}")

    train_ds = ArtifactPairs(train["frames"], train["masks"], train["opening"], seed=args.seed)
    val_ds = ArtifactPairs(val["frames"], val["masks"], val["opening"], seed=10_000, fixed=True)
    test_ds = ArtifactPairs(test["frames"], test["masks"], test["opening"], seed=20_000, fixed=True)

    # Fixed test artifacts so every method sees exactly the same inputs.
    noisy = np.stack([test_ds[i][0][0].numpy() for i in range(len(test_ds))])
    clean, masks = test["frames"], test["masks"]

    restored, runtime, params, histories = {}, {}, {}, {}
    for m in CLASSICAL:
        restored[m], runtime[m] = restore_all(m, noisy)
        params[m] = 0
    for m in args.models:
        print(f"training {m} ...")
        res = train_model(m, train_ds, val_ds, out / "checkpoints", args.epochs,
                          args.batch_size, seed=args.seed, device=device)
        histories[m] = res["history"]
        params[m] = res["params"]
        restored[m], runtime[m] = restore_all(m, noisy, res["model"], device)

    # Upper bound: segmentation on the artifact-free frames.
    restored_ref = dict(restored, clean_reference=clean)
    runtime["clean_reference"], params["clean_reference"] = 0.0, 0
    frames = pd.concat([per_frame_metrics(m, restored_ref[m], clean, masks) for m in restored_ref])
    use_gaw = args.data == "synthetic"
    gaws = pd.concat([gaw_table(m, frames[frames.method == m], masks, test["seq_id"])
                      for m in restored_ref]) if use_gaw else None

    summary = summarise(frames, gaws, {"ms_per_frame": runtime, "params": params})
    best_dl = summary.loc[summary.index.isin(args.models)].index[0]
    best_cl = summary.loc[summary.index.isin(CLASSICAL)].index[0]
    test_res = paired_test(frames[frames.gt_area > 0], best_dl, best_cl, "dice")

    frames.to_csv(out / "per_frame_metrics.csv", index=False)
    if use_gaw:
        gaws.to_csv(out / "gaw_per_sequence.csv", index=False)
    summary.round(4).to_csv(out / "summary.csv")
    (out / "stats.json").write_text(json.dumps(test_res, indent=2))

    plot_examples(out, noisy, clean, masks, restored)
    plot_curves(out, histories)
    if use_gaw:
        plot_gaw(out, test, restored, ["identity", best_cl, best_dl])

    pd.set_option("display.width", 140)
    print("\n=== summary (sorted by Dice) ===")
    print(summary.round(3).to_string())
    print(f"\nWilcoxon (Dice on open-glottis frames) {best_dl} vs {best_cl}: "
          f"median diff {test_res['median_diff']:.3f}, p = {test_res['p_value']:.2e}")
    print(f"\nresults written to {out.resolve()}")


if __name__ == "__main__":
    main()
