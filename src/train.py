"""Training loop for the restoration models."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import build_model, n_params


def train_model(
    name: str,
    train_ds,
    val_ds,
    out_dir: Path,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Train one model with L1 loss, keep the checkpoint with the best val loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best, history = float("inf"), []
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{name}.pt"
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for noisy, clean, _ in train_dl:
            noisy, clean = noisy.to(device), clean.to(device)
            loss = F.l1_loss(model(noisy), clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(noisy)
        sched.step()
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for noisy, clean, _ in val_dl:
                noisy, clean = noisy.to(device), clean.to(device)
                va_loss += F.l1_loss(model(noisy), clean).item() * len(noisy)
        va_loss /= len(val_ds)
        history.append({"epoch": ep, "train_l1": tr_loss, "val_l1": va_loss})
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), ckpt)
        print(f"  [{name}] epoch {ep:02d}/{epochs}  train L1 {tr_loss:.4f}  val L1 {va_loss:.4f}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    return {
        "model": model.eval(),
        "history": history,
        "params": n_params(model),
        "train_seconds": time.time() - t0,
    }
