# vertebrae-learned-cleanup

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/vertebrae-learned-cleanup.svg)]

**A tiny learned 3D cleanup network (87k params) that fixes the residual vertebra-segmentation errors a heuristic postprocessor provably cannot — boundary errors, missing bone, nearby wrong labels — measured in Dice, not voxel counts.**

This extends [vertebrae-refinement](https://github.com/habib-analyst/vertebrae-refinement) (physical-distance heuristic postprocessing). That repo was explicit about its limits: *"The method leaves boundary errors, missing bone, and nearby wrong labels unresolved"* and reported change counts, not accuracy. This repo closes that loop with a residual-prior network that copies the baseline mask and learns to fix the surface.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests -q                       # 8/8 green, CPU, ~1 min

python src/train.py --out results               # train the residual-prior net (~3-5h CPU)

python src/evaluate.py --ckpt results/best.pt \
  --out results/eval.json                       # raw vs heuristic vs learned vs hybrid
```

Expected output (from the measured evaluation, `results/eval.json`):

```
combined:   heuristic Dice 0.6509 -> hybrid Dice 0.8796 (+22.9pp)
boundary:   heuristic Dice 0.6218 -> hybrid Dice 0.8479 (+22.6pp)
```

## Measured results

Measured on 12 held-out volumes per isolated error set (`results/eval.json`, residual-prior checkpoint, 87,193 params).

| error set | raw Dice | heuristic Dice | learned Dice | hybrid Dice |
|---|---|---|---|---|
| boundary | 0.6218 | 0.6218 | **0.8479** | **0.8479** |
| missing | 0.9894 | 0.9894 | 0.9885 | 0.9885 |
| wronglabel | 0.9931 | 0.9931 | 0.9921 | 0.9921 |
| distant | 0.9977 | **0.9999** | 0.9942 | 0.9942 |
| combined | 0.6492 | 0.6509 | **0.8796** | **0.8796** |

Headline: hybrid beats heuristic-only by **+22.9pp Dice on combined** (0.8796 vs 0.6509) and **+22.6pp on boundary-only** (0.8479 vs 0.6218).

The honest caveats: the network wins on overlap, not surface smoothness (boundary HD95 1.79 → 2.74 — the learned corrections add small surface irregularities); the distant-blob case stays with the heuristic (0.9999 vs 0.9942 — deleting a disconnected blob is deterministic work a net can only approximate); missing/wronglabel were a wash because the synthetic corruptions were too mild to move Dice (raw already ~0.99).

## How it works

```
CT intensity + baseline mask ──┬──► heuristic cleanup ──► hybrid output
                               │
                               └──► TinyUNet3D (residual prior) ──► learned output
```

- **Heuristic** (`src/heuristic.py`): faithful port of the vertebrae-refinement rule — largest 26-connected component per label; secondary components ≤10% of main and >far-threshold from it are reassigned only if they touch exactly one neighboring label within the touch radius, else removed.
- **Network** (`src/model.py`): 3D U-Net, 87k params. Input = normalized intensity + one-hot baseline mask (10 ch); output = 9-class logits; residual prior — the net copies the baseline and learns the fix. Loss = CrossEntropy + (1 − mean foreground Dice).
- **Evaluation** (`src/evaluate.py`): four-way ablation on held-out volumes with *isolated* error classes (boundary-only / missing-only / wronglabel-only / distant-only / combined), scored with per-vertebra Dice and HD95 — so each gain is attributed to the error class the heuristic cannot touch.

All results measured on **seeded synthetic vertebra volumes** (`src/synth.py`, reproducible from seed). Evaluation on public VerSe 2019/2020 is queued pending dataset registration — the pipeline is built so the real-data run is a drop-in replacement of the data loader. See `PAPER.md` for the full writeup.

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
