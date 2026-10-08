"""Synthetic vertebra-like 3D volumes for Project #7 (vertebrae-learned-cleanup).

Generates ground-truth label maps (background=0, vertebrae=1..N) plus a
CT-like intensity channel, then a *corrupted baseline prediction* that mimics
real SuPreM/SwinUNETR failure modes:

  - boundary noise      (morphological jitter at vertebra surfaces)
  - missing bone        (chunks carved out of a vertebra body)
  - wrong nearby label  (fragment of vertebra i relabeled as neighbor i+/-1)
  - distant FP blob     (small component far away -- the ONE error class the
                         heuristic postprocessor in vertebrae-refinement fixes)

Each corrupted volume is tagged with which error types were injected so the
failure-mode analysis can attribute Dice gains per error class.

Everything is seeded and deterministic given `seed`.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

N_VERT = 8          # vertebrae per synthetic volume (keeps CPU training fast)
SHAPE = (64, 64, 64)
VOXEL_MM = 1.0      # isotropic 1mm, so repo thresholds (50mm/20mm) map 1:1


def _ellipsoid(shape, center, radii):
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    d = (((xx - center[0]) / radii[0]) ** 2
         + ((yy - center[1]) / radii[1]) ** 2
         + ((zz - center[2]) / radii[2]) ** 2)
    return d <= 1.0


def make_ground_truth(rng: np.random.Generator, shape=SHAPE, n_vert=N_VERT):
    """Return (label_map[int], intensity[float32])."""
    labels = np.zeros(shape, dtype=np.int64)
    intensity = rng.normal(20.0, 8.0, shape).astype(np.float32)  # soft tissue

    zc = np.linspace(9, shape[0] - 9, n_vert)
    xc = shape[2] / 2 + 5.0 * np.sin(np.linspace(0, 1.1 * np.pi, n_vert)
                                    + rng.uniform(0, 2 * np.pi))
    yc = shape[1] / 2 + rng.normal(0, 1.2, n_vert)

    for i in range(n_vert):
        rx = rng.uniform(6.5, 8.5)
        ry = rng.uniform(5.5, 7.0)
        rz = rng.uniform(4.0, 5.2)
        body = _ellipsoid(shape, (xc[i], yc[i], zc[i]), (rx, ry, rz))
        # spinous process: posterior bump (makes boundaries non-trivial)
        bump = _ellipsoid(shape,
                          (xc[i], yc[i] - ry - 1.5, zc[i]),
                          (rx * 0.35, 3.0, rz * 0.5))
        vert = body | bump
        labels[vert] = i + 1
        # bone intensity with trabecular-ish texture
        tex = rng.normal(0, 25, shape).astype(np.float32)
        tex = ndi.gaussian_filter(tex, 1.0)
        intensity[vert] = 420.0 + tex[vert] + rng.normal(0, 30, vert.sum())
    # clip to plausible HU-ish range
    intensity = np.clip(intensity, -100, 1200).astype(np.float32)
    return labels, intensity


def _random_ball(rng, shape, center, radius):
    return _ellipsoid(shape, center, (radius, radius, radius))


def corrupt_baseline(labels: np.ndarray, rng: np.random.Generator,
                     error_types=("boundary", "missing", "wronglabel", "distant")):
    """Simulate a SuPreM-style imperfect prediction. Returns (pred, tags)."""
    pred = labels.copy()
    tags = []
    shape = labels.shape
    present = [l for l in range(1, N_VERT + 1) if (labels == l).any()]

    if "boundary" in error_types and present:
        # morphological jitter: randomly erode or dilate each vertebra surface
        for l in present:
            m = (labels == l)
            if rng.random() < 0.5:
                m2 = ndi.binary_erosion(m, iterations=int(rng.integers(1, 3)))
            else:
                m2 = ndi.binary_dilation(m, iterations=int(rng.integers(1, 3)))
            pred[m & ~m2] = 0
            pred[m2 & (pred == 0)] = l
        tags.append("boundary")

    if "missing" in error_types and present:
        # carve 1-2 chunks out of 1-2 random vertebrae (missing bone)
        for _ in range(int(rng.integers(1, 3))):
            l = present[int(rng.integers(len(present)))]
            vox = np.argwhere(labels == l)
            c = vox[int(rng.integers(len(vox)))]
            r = float(rng.uniform(2.5, 4.5))
            chunk = _random_ball(rng, shape, (c[2], c[1], c[0]), r)
            pred[chunk & (labels == l)] = 0
        tags.append("missing")

    if "wronglabel" in error_types and len(present) > 1:
        # relabel a fragment of vertebra i as neighbor i+/-1 (nearby wrong label)
        l = present[int(rng.integers(len(present)))]
        neigh = l + (1 if (l < N_VERT and rng.random() < 0.5) or l == 1 else -1)
        vox = np.argwhere(labels == l)
        c = vox[int(rng.integers(len(vox)))]
        r = float(rng.uniform(2.0, 3.5))
        frag = _random_ball(rng, shape, (c[2], c[1], c[0]), r)
        pred[frag & (labels == l)] = neigh
        tags.append("wronglabel")

    if "distant" in error_types and present:
        # small false-positive blobs far from any vertebra (heuristic-fixable).
        # Placed at >=22mm from every vertebra main centroid so they qualify
        # under the scaled (18mm) far-threshold used in the synthetic eval.
        cents = {}
        for l in present:
            m = (labels == l)
            lab, n = ndi.label(m, structure=ndi.generate_binary_structure(3, 3))
            sizes = ndi.sum(m, lab, range(1, n + 1))
            c = ndi.center_of_mass(m, lab, range(1, n + 1))[int(np.argmax(sizes))]
            cents[l] = np.asarray(c, dtype=float)
        placed = 0
        for _ in range(40):
            if placed >= int(rng.integers(1, 4)):
                break
            c = np.array([rng.uniform(0, shape[2]), rng.uniform(0, shape[1]),
                          rng.uniform(0, shape[0])])
            if min(np.linalg.norm(c - v) for v in cents.values()) < 22.0:
                continue
            r = float(rng.uniform(1.2, 2.2))
            blob = _random_ball(rng, shape, c, r)
            blob = blob & (labels == 0)
            if blob.sum() > 8:
                pred[blob] = present[int(rng.integers(len(present)))]
                placed += 1
        tags.append("distant")

    return pred.astype(np.int64), tags


def make_volume(seed: int, error_types=("boundary", "missing", "wronglabel", "distant")):
    rng = np.random.default_rng(seed)
    gt, intensity = make_ground_truth(rng)
    pred, tags = corrupt_baseline(gt, rng, error_types)
    return {"gt": gt, "intensity": intensity, "baseline": pred, "tags": tags}


def one_hot(labels: np.ndarray, n_classes: int = N_VERT + 1):
    oh = np.zeros((n_classes,) + labels.shape, dtype=np.float32)
    for c in range(n_classes):
        oh[c] = (labels == c)
    return oh
