# Vertebrae Learned Cleanup

A tiny 3D cleanup network that fixes the residual segmentation errors a
heuristic postprocessor **provably cannot** — boundary errors, missing bone,
and nearby wrong labels — measured in Dice, not voxel counts.

This project extends
[vertebrae-refinement](https://github.com/habib-analyst/vertebrae-refinement)
(physical-distance heuristic postprocessing for vertebra masks). That repo was
explicit about its limits: *"The method leaves boundary errors, missing bone,
and nearby wrong labels unresolved"* and reported change counts, not accuracy.
This repo closes that loop.

## What it does

```
CT intensity + baseline mask ──┬──► heuristic cleanup ──► hybrid output
                               │
                               └──► TinyUNet3D ──► learned-only output
```

Four-way ablation on held-out volumes: `raw` (no cleanup) vs `heuristic`
vs `learned` vs `hybrid` (heuristic → network), scored with per-vertebra
**Dice** and **HD95**.

## Key result (measured)

See [`results/eval.json`](results/eval.json) and `PAPER.md`. Headline numbers
are reproduced here after evaluation — no claims without measurements.

## Data honesty

Current results are measured on **seeded synthetic vertebra volumes**
(`src/synth.py`): ellipsoid vertebral bodies with trabecular-like texture,
corrupted with the four SuPreM-style failure modes (boundary jitter, carved-out
bone, neighbor-label flips, distant false-positive blobs). Every volume is
reproducible from its seed.

Evaluation on public **VerSe 2019/2020** data is queued pending dataset
registration — the pipeline is built so the real-data run is a drop-in
replacement of the data loader. See PAPER.md §6 (Limits).

## Method notes

- **Heuristic** (`src/heuristic.py`): faithful port of the vertebrae-refinement
  rule — largest 26-connected component per label; secondary components ≤10%
  of main and >far-threshold from it are reassigned only if they touch exactly
  one neighboring label within the touch radius, else removed. Thresholds are
  parameters defaulting to the repo's 50/20mm; the synthetic eval scales them
  ×0.36 to match the ~1/3-scale synthetic anatomy (same algorithm).
- **Network** (`src/model.py`): 3D U-Net, 87k params. Input = normalized
  intensity + one-hot baseline mask (10 ch); output = 9-class logits.
  Loss = CrossEntropy + (1 − mean foreground Dice).
- **Failure-mode analysis** (`src/evaluate.py`): test volumes generated with
  *isolated* error types, so Dice gains are attributed per error class —
  this is what shows the network fixing exactly the errors the heuristic
  cannot.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# tests (CPU, ~1 min)
python -m pytest tests -q

# train (~3-5h CPU; GPU optional, set CUDA_VISIBLE_DEVICE if available)
python src/train.py --out results

# evaluate: raw vs heuristic vs learned vs hybrid + failure-mode breakdown
python src/evaluate.py --ckpt results/best.pt --out results/eval.json
```

## Layout

```
src/
  synth.py      synthetic vertebra volumes + SuPreM-style corruption
  heuristic.py  vertebrae-refinement heuristic (ported, thresholds parametrized)
  model.py      TinyUNet3D (87k params)
  train.py      training loop (CE + Dice loss)
  evaluate.py   4-way ablation + per-error-class failure analysis
tests/          7 unit tests, all green on CPU
PAPER.md        full writeup: claim, method, measured results, limits
```

## License

MIT. Synthetic-data code is original; the heuristic is a clean-room port of
the author's own MIT-licensed vertebrae-refinement repo.
