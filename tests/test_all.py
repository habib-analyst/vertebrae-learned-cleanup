"""Unit tests for vertebrae-learned-cleanup (CPU, fast)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from synth import make_volume, one_hot, N_VERT  # noqa: E402
from heuristic import heuristic_cleanup  # noqa: E402
from model import TinyUNet3D, count_params  # noqa: E402
from evaluate import dice_per_class  # noqa: E402
from evaluate import learned_predict  # noqa: E402


def test_residual_prior_copies_baseline():
    # A zero-output network must reproduce the baseline mask exactly:
    # the residual prior makes "copy the baseline" the default behavior,
    # so training only ever learns corrections.
    v = make_volume(31, error_types=("boundary", "missing"))
    net = TinyUNet3D()
    net.eval()
    with torch.no_grad():
        net.head.weight.zero_()
        net.head.bias.zero_()
    pred = learned_predict(net, v["intensity"], v["baseline"])
    assert np.array_equal(pred, v["baseline"])


def test_synth_deterministic():
    a = make_volume(7)
    b = make_volume(7)
    assert np.array_equal(a["gt"], b["gt"])
    assert np.array_equal(a["baseline"], b["baseline"])
    assert a["gt"].shape == (64, 64, 64)
    assert set(np.unique(a["gt"])) <= set(range(N_VERT + 1))
    assert (a["gt"] > 0).sum() > 1000  # there is actual bone


def test_synth_errors_present():
    v = make_volume(11)
    assert not np.array_equal(v["gt"], v["baseline"])  # corruption happened
    assert set(v["tags"]) == {"boundary", "missing", "wronglabel", "distant"}


def test_one_hot():
    v = make_volume(13)
    oh = one_hot(v["baseline"], N_VERT + 1)
    assert oh.shape == (N_VERT + 1, 64, 64, 64)
    assert np.allclose(oh.sum(0), 1.0)


def test_heuristic_removes_distant_fp():
    # distant FP blobs must be removed; main components untouched
    v = make_volume(21, error_types=("distant",))
    gt, base = v["gt"], v["baseline"]
    fp_before = int(((base > 0) & (gt == 0)).sum())
    assert fp_before > 0, "test needs distant FPs to exist"
    cleaned = heuristic_cleanup(base, far_mm=18.0, touch_mm=7.0)
    fp_after = int(((cleaned > 0) & (gt == 0)).sum())
    assert fp_after < fp_before
    # no background voxel may become foreground that wasn't already
    assert ((cleaned > 0) & (base == 0) & (gt == 0)).sum() == 0


def test_heuristic_does_not_fix_boundary():
    v = make_volume(22, error_types=("boundary",))
    cleaned = heuristic_cleanup(v["baseline"], far_mm=18.0, touch_mm=7.0)
    d_raw = dice_per_class(v["baseline"], v["gt"]).mean()
    d_heu = dice_per_class(cleaned, v["gt"]).mean()
    # heuristic is not designed for boundary errors: gain must be ~zero
    assert abs(d_heu - d_raw) < 0.01, (d_raw, d_heu)


def test_model_forward():
    net = TinyUNet3D()
    x = torch.randn(1, N_VERT + 2, 64, 64, 64)  # intensity + 9 one-hot = 10 ch
    y = net(x)
    assert y.shape == (1, N_VERT + 1, 64, 64, 64)
    assert count_params(net) < 200_000


def test_dice_perfect_and_empty():
    g = np.zeros((8, 8, 8), np.int64)
    g[2:5, 2:5, 2:5] = 1
    d = dice_per_class(g, g, n_classes=2)
    assert abs(d[0] - 1.0) < 1e-9
    d2 = dice_per_class(np.zeros_like(g), g, n_classes=2)
    assert d2[0] == 0.0
