"""Evaluation: heuristic-only vs learned-only vs hybrid, with failure-mode analysis.

Test volumes are generated with ISOLATED error types so Dice gains can be
attributed per error class:
  - boundary-only, missing-only, wronglabel-only, distant-only, all-combined

Metrics per volume: mean foreground Dice per vertebra, and HD95 per vertebra
(95th-percentile Hausdorff, robust to outliers).
Methods:
  raw        = corrupted baseline prediction (no cleanup)
  heuristic  = vertebrae-refinement heuristic (heuristic.py)
  learned    = TinyUNet3D applied to the raw baseline
  hybrid     = heuristic first, then TinyUNet3D on the heuristic output
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(__file__))
from synth import make_volume, one_hot, N_VERT  # noqa: E402
from heuristic import heuristic_cleanup  # noqa: E402
from model import TinyUNet3D  # noqa: E402

N_CLASSES = N_VERT + 1

# Heuristic thresholds scaled x0.36 to the synthetic anatomy (~1/3-scale
# vertebrae vs real CT): same algorithm as vertebrae-refinement, thresholds
# matched to anatomy size. Documented in PAPER.md.
HEU_FAR_MM = 18.0
HEU_TOUCH_MM = 7.0
ERROR_SETS = {
    "boundary": ("boundary",),
    "missing": ("missing",),
    "wronglabel": ("wronglabel",),
    "distant": ("distant",),
    "combined": ("boundary", "missing", "wronglabel", "distant"),
}


def dice_per_class(pred, gt, n_classes=N_CLASSES):
    out = []
    for c in range(1, n_classes):
        p, g = (pred == c), (gt == c)
        if not g.any() and not p.any():
            out.append(1.0)
            continue
        inter = (p & g).sum()
        out.append(2 * inter / (p.sum() + g.sum() + 1e-9))
    return np.array(out)


def hd95_per_class(pred, gt, n_classes=N_CLASSES, voxel_mm=1.0):
    out = []
    for c in range(1, n_classes):
        p, g = (pred == c), (gt == c)
        if not g.any() or not p.any():
            out.append(float("nan"))
            continue
        dt_g = ndi.distance_transform_edt(~g, sampling=voxel_mm)
        dt_p = ndi.distance_transform_edt(~p, sampling=voxel_mm)
        hd = max(np.percentile(dt_g[p], 95), np.percentile(dt_p[g], 95))
        out.append(hd)
    return np.array(out)


@torch.no_grad()
def learned_predict(net, intensity, baseline_mask):
    from train import RESID_W  # noqa: E402  (same residual prior as training)
    inten = np.clip((intensity.astype(np.float32) - 200.0) / 400.0, -1.5, 2.5)
    x = np.concatenate([inten[None], one_hot(baseline_mask, N_CLASSES)], 0)
    x = torch.from_numpy(x)[None]
    logits = net(x) + RESID_W * x[:, 1:, :, :, :]
    return logits.argmax(1)[0].numpy().astype(np.int64)


def evaluate(ckpt, test_start=3000, n_test=12):
    net = TinyUNet3D()
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()

    rows = []
    for err_name, err_types in ERROR_SETS.items():
        dices = {m: [] for m in ("raw", "heuristic", "learned", "hybrid")}
        hds = {m: [] for m in ("raw", "heuristic", "learned", "hybrid")}
        for i in range(n_test):
            v = make_volume(test_start + i * 10 + list(ERROR_SETS).index(err_name),
                            error_types=err_types)
            gt = v["gt"]
            raw = v["baseline"]
            heu = heuristic_cleanup(raw, far_mm=HEU_FAR_MM, touch_mm=HEU_TOUCH_MM)
            lea = learned_predict(net, v["intensity"], raw)
            hyb = learned_predict(net, v["intensity"], heu)
            for name, pr in (("raw", raw), ("heuristic", heu),
                             ("learned", lea), ("hybrid", hyb)):
                d = dice_per_class(pr, gt)
                h = hd95_per_class(pr, gt)
                dices[name].append(d.mean())
                hds[name].append(np.nanmean(h))
        rows.append({
            "error_set": err_name,
            "n_volumes": n_test,
            "mean_dice": {m: float(np.mean(dices[m])) for m in dices},
            "std_dice": {m: float(np.std(dices[m])) for m in dices},
            "mean_hd95_mm": {m: float(np.nanmean(hds[m])) for m in hds},
        })
        r = rows[-1]
        print(f"[{err_name}] dice raw={r['mean_dice']['raw']:.4f} "
              f"heu={r['mean_dice']['heuristic']:.4f} "
              f"learned={r['mean_dice']['learned']:.4f} "
              f"hybrid={r['mean_dice']['hybrid']:.4f} | "
              f"hd95 heu={r['mean_hd95_mm']['heuristic']:.2f} "
              f"hybrid={r['mean_hd95_mm']['hybrid']:.2f}", flush=True)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/best.pt")
    p.add_argument("--out", default="results/eval.json")
    p.add_argument("--n-test", type=int, default=12)
    args = p.parse_args()
    rows = evaluate(args.ckpt, n_test=args.n_test)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {args.out}")
