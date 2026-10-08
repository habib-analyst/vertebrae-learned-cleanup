"""Training for the 3D cleanup network.

Dataset: synthetic volumes from synth.py. Input channels: normalized intensity
+ one-hot baseline mask. Target: ground-truth label map.
Loss: CrossEntropy + (1 - mean Dice over foreground classes).
Augmentation: axis flips + small intensity noise (label-preserving).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from synth import make_volume, one_hot, N_VERT  # noqa: E402
from model import TinyUNet3D, count_params  # noqa: E402

N_CLASSES = N_VERT + 1

# Residual prior: the network predicts a *correction* added to a scaled copy
# of the baseline one-hot mask (channels 1.. of x). At init the model copies
# the baseline (already ~99% right on most volumes) and training only learns
# the fixes. Without this, the foreground classes are 3% of voxels and the
# network collapses to the "predict background everywhere" minimum.
RESID_W = 5.0


def forward_with_prior(net, x):
    return net(x) + RESID_W * x[:, 1:, :, :, :]


class CleanupDataset(Dataset):
    def __init__(self, seeds, augment=False):
        self.seeds = seeds
        self.augment = augment

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, i):
        v = make_volume(int(self.seeds[i]))
        inten = v["intensity"].astype(np.float32)
        inten = np.clip((inten - 200.0) / 400.0, -1.5, 2.5)  # normalize
        x = np.concatenate([inten[None], one_hot(v["baseline"], N_CLASSES)], 0)
        y = v["gt"].astype(np.int64)
        if self.augment:
            rng = np.random.default_rng(int(self.seeds[i]) + 999)
            for ax in (1, 2, 3):  # flip spatial axes (x is channel dim)
                if rng.random() < 0.5:
                    x = np.flip(x, ax).copy()
                    y = np.flip(y, ax - 1).copy()
            x[0] += rng.normal(0, 0.02, x[0].shape).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


def dice_loss(logits, target, eps=1e-6):
    prob = torch.softmax(logits, 1)
    tgt = torch.nn.functional.one_hot(target, N_CLASSES).permute(0, 4, 1, 2, 3).float()
    inter = (prob * tgt).sum((0, 2, 3, 4))
    union = prob.sum((0, 2, 3, 4)) + tgt.sum((0, 2, 3, 4))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice[1:].mean()  # foreground classes only


def train(args):
    torch.manual_seed(0)
    np.random.seed(0)
    train_ds = CleanupDataset(range(args.train_start, args.train_start + args.n_train),
                              augment=True)
    val_ds = CleanupDataset(range(args.val_start, args.val_start + args.n_val))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=1, num_workers=0)

    net = TinyUNet3D()
    print(f"params: {count_params(net):,}", flush=True)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val = float("inf")
    log = []
    start_ep = 0
    ckpt = os.path.join(args.out, "last.pt")
    if getattr(args, "resume", False) and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        best_val = state["best_val"]
        start_ep = state["epoch"] + 1
        log = state["log"]
        print(f"resumed from epoch {state['epoch']} (best_val={best_val:.4f})",
              flush=True)
    for ep in range(start_ep, args.epochs):
        t0 = time.time()
        net.train()
        tr_loss = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            logits = forward_with_prior(net, x)
            loss = ce(logits, y) + dice_loss(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_dl)
        net.eval()
        va_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                logits = forward_with_prior(net, x)
                va_loss += (ce(logits, y) + dice_loss(logits, y)).item()
        va_loss /= len(val_dl)
        dt = time.time() - t0
        log.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                    "sec": dt})
        print(f"ep {ep:02d} train={tr_loss:.4f} val={va_loss:.4f} ({dt:.0f}s)",
              flush=True)
        if va_loss < best_val:
            best_val = va_loss
            torch.save(net.state_dict(), os.path.join(args.out, "best.pt"))
        torch.save({"epoch": ep, "net": net.state_dict(), "opt": opt.state_dict(),
                    "best_val": best_val, "log": log}, ckpt)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(f"best val loss: {best_val:.4f}")
    open(os.path.join(args.out, "TRAIN_DONE"), "w").write("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-start", type=int, default=1000)
    p.add_argument("--n-train", type=int, default=36)
    p.add_argument("--val-start", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--resume", action="store_true",
                   help="resume from out/last.pt if present")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)
