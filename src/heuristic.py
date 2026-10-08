"""Heuristic postprocessor — faithful port of the vertebrae-refinement method.

From habib-analyst/vertebrae-refinement (README "Method"):
  1. Find the largest 26-connected component for each vertebra label.
  2. A secondary component is a candidate only if it has at most 10% as many
     voxels as the main component AND is more than 50mm from the main component
     (world coordinates via the NIfTI affine; here isotropic 1mm voxels).
  3. Reassign the component only if it touches exactly one plausible retained
     label whose main component is within 20mm. Otherwise remove it (→0).

What it provably cannot fix (documented in that repo's README): boundary
errors, missing bone, nearby wrong labels. Those are the residual errors the
learned cleanup network targets.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

STRUCT26 = ndi.generate_binary_structure(3, 3)  # 26-connectivity

MAX_FRAC = 0.10   # secondary component at most 10% of main
FAR_MM = 50.0     # candidate must be >50mm from main component
TOUCH_MM = 20.0   # reassignment contact radius


def _component_centroids(mask):
    lab, n = ndi.label(mask, structure=STRUCT26)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    cents = ndi.center_of_mass(mask, lab, range(1, n + 1))
    return lab, n, np.asarray(sizes), np.asarray(cents)


def heuristic_cleanup(pred: np.ndarray, voxel_mm: float = 1.0,
                      n_labels: int = 8, far_mm: float = FAR_MM,
                      touch_mm: float = TOUCH_MM) -> np.ndarray:
    """Apply the vertebrae-refinement heuristic. Returns cleaned label map.

    far_mm/touch_mm default to the repo's 50/20mm. For synthetic anatomies
    built at ~1/3 real-vertebra scale, pass scaled values (e.g. 18/7mm) --
    same algorithm, thresholds matched to anatomy size.
    """
    out = pred.copy()
    # main component mask per label (for the "touches exactly one label" test)
    main_masks = {}
    for l in range(1, n_labels + 1):
        m = (pred == l)
        if not m.any():
            continue
        lab, n, sizes, _ = _component_centroids(m)
        main_masks[l] = (lab == (np.argmax(sizes) + 1))

    for l in range(1, n_labels + 1):
        m = (pred == l)
        if not m.any():
            continue
        lab, n, sizes, cents = _component_centroids(m)
        if n <= 1:
            continue
        order = np.argsort(-sizes)
        main_id = order[0] + 1
        main_size = sizes[order[0]]
        main_cent = np.asarray(cents[order[0]]) * voxel_mm
        for cid in order[1:]:
            comp = (lab == (cid + 1))
            if sizes[cid] > MAX_FRAC * main_size:
                continue  # substantial secondary component: leave intact
            dist = float(np.linalg.norm(
                np.asarray(cents[cid]) * voxel_mm - main_cent))
            if dist <= far_mm:
                continue  # nearby fragment: leave intact
            # candidate: try reassignment to exactly one neighboring label
            dilated = ndi.binary_dilation(comp, iterations=int(touch_mm / voxel_mm))
            touched = set()
            for l2, mm in main_masks.items():
                if l2 == l:
                    continue
                if (dilated & mm).any():
                    # main component of l2 within touch radius?
                    d2 = ndi.distance_transform_edt(~mm, sampling=voxel_mm)
                    if d2[comp].min() <= touch_mm:
                        touched.add(l2)
            if len(touched) == 1:
                out[comp] = next(iter(touched))
            else:
                out[comp] = 0
    return out
