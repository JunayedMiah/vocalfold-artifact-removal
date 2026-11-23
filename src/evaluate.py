"""Evaluation: image fidelity, downstream segmentation, GAW, runtime, statistics."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .models import classical_restore
from .segment import dice, gaw, gaw_metrics, iou, segment_glottis


def restore_all(method: str, noisy: np.ndarray, model=None, device="cpu", batch=32):
    """Restore a stack (N, H, W). Returns (restored, ms_per_frame)."""
    t0 = time.perf_counter()
    if model is None:
        out = np.stack([classical_restore(x, method) for x in noisy])
    else:
        outs = []
        with torch.no_grad():
            for i in range(0, len(noisy), batch):
                x = torch.from_numpy(noisy[i:i + batch])[:, None].to(device)
                outs.append(model(x)[:, 0].cpu().numpy())
        out = np.concatenate(outs)
    ms = 1000 * (time.perf_counter() - t0) / len(noisy)
    return out.astype(np.float32), ms


def per_frame_metrics(method, restored, clean, masks) -> pd.DataFrame:
    rows = []
    for i in range(len(restored)):
        pred = segment_glottis(restored[i])
        psnr = (np.nan if np.array_equal(clean[i], restored[i])   # reference row: PSNR undefined
                else peak_signal_noise_ratio(clean[i], restored[i], data_range=1.0))
        rows.append({
            "method": method,
            "frame": i,
            "psnr": psnr,
            "ssim": structural_similarity(clean[i], restored[i], data_range=1.0),
            "dice": dice(pred, masks[i]),
            "iou": iou(pred, masks[i]),
            "pred_area": float(pred.sum()),
            "gt_area": float(masks[i].sum()),
        })
    return pd.DataFrame(rows)


def gaw_table(method, frame_df, masks, seq_id) -> pd.DataFrame:
    """GAW metrics per sequence (only meaningful for real sequences)."""
    rows = []
    gt_area = gaw(masks)
    for s in np.unique(seq_id):
        idx = np.where(seq_id == s)[0]
        pred_area = frame_df.loc[frame_df.frame.isin(idx), "pred_area"].to_numpy()
        rows.append({"method": method, "sequence": int(s),
                     **gaw_metrics(pred_area, gt_area[idx])})
    return pd.DataFrame(rows)


def summarise(frames: pd.DataFrame, gaws: pd.DataFrame | None, extra: dict) -> pd.DataFrame:
    s = frames.groupby("method")[["psnr", "ssim", "dice", "iou"]].mean()
    # Closed frames are trivial (empty vs empty = 1.0); report open frames separately.
    s["dice_open"] = frames[frames.gt_area > 0].groupby("method")["dice"].mean()
    if gaws is not None and len(gaws):
        s = s.join(gaws.groupby("method")[["gaw_nmae", "gaw_pearson"]].mean())
    for col, values in extra.items():
        s[col] = pd.Series(values)
    return s.sort_values("dice_open", ascending=False)


def paired_test(frames: pd.DataFrame, a: str, b: str, metric: str = "dice") -> dict:
    """Wilcoxon signed-rank test on per-frame scores of two methods."""
    x = frames[frames.method == a].sort_values("frame")[metric].to_numpy()
    y = frames[frames.method == b].sort_values("frame")[metric].to_numpy()
    diff = x - y
    if np.allclose(diff, 0):
        return {"a": a, "b": b, "metric": metric, "median_diff": 0.0, "p_value": 1.0}
    stat, p = wilcoxon(x, y)
    return {"a": a, "b": b, "metric": metric,
            "median_diff": float(np.median(diff)), "p_value": float(p)}
