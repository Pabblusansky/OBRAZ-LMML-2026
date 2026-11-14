# The 300-Year Old Tiger Nest Painting

| Field | Value |
|---|---|
| Country | Bhutan |
| Difficulty | Medium |
| Category | computer vision — image strip reassembly (unshredding) |
| Solved by | Stanislav |
| Status | solved |
| Flag | `SIGMOID_...` (fill in after the platform accepts it) |

## 60-second version

A 1536×1024 painting was cut into 192 vertical strips (8 px wide) and shuffled. Neighbouring
strips continue each other, so for every pair (A, B) we predict B's first column from A's last two
columns (and vice versa) and measure the error. Costs are normalised ("how much better than the
alternatives"), strips are chained greedily from the most convincing join, and the chain is cut at
its least convincing join so it starts at the true left edge. Train image: perfect order, SSIM 1.0.

## Task

Reassemble shuffled vertical strips of an image. `main.py` reads strips from a folder named
`input` and writes `reconstructed_image.png` + `reconstructed_order.json` next to itself.
Pass = SSIM ≥ 0.90, order match ≥ 0.90 (mirror accepted), runtime < 20 s.

## Approach

```bash
python main.py                          # reads ./input next to main.py
python check_reconstruction.py          # local scores on the train image (development only)
python check_reconstruction.py --stress # + noisy strips and other photos cut into strips
```

1. **Join cost** `cost[i, j]` (strip j right after strip i): prediction error across the cut,
   both directions: `|2·lastA − secondLastA − firstB| + |2·firstB − secondB − lastA|`, summed over
   all pixels and colour channels → 192×192 matrix.
2. **Normalise**: divide each cost by the best alternative join for the same two strips.
3. **Greedy chaining**: accept joins from most to least convincing, if the left strip has no right
   neighbour yet, the right strip no left neighbour yet, and no loop is closed (union-find).
4. **Ring cut**: close the chain into a ring and cut at the least convincing join.

Tested with Python 3.13, numpy 2.5.3, opencv 5.0.0 (the pinned evaluator versions).

## What failed / what we learned

| # | Idea | Result | Lesson |
|---|---|---|---|
| 1 | plain edge difference vs prediction vs MGC (gradient-based) | all 191/191 correct joins on train | the painting is richly textured; chose prediction (robust, cheap) |
| 2 | stress test: other photos cut into strips, raw costs | 56/60 pass | failures were a **wrap-around**: right edge matched left edge, chain started mid-image |
| 3 | normalised costs + ring cut | 58/60 pass | remaining failures: photos with blank borders (identical strips cannot be ordered) |
| 4 | noise up to σ=20 on every strip | still perfect | |

## Files

- `main.py` — the submission (self-contained).
- `check_reconstruction.py` — local scoring + stress tests (needs scikit-image for SSIM).
- `requirements.txt` — pinned versions given by the task.
- `task.md` — original task text. Data in `data/` (gitignored).
