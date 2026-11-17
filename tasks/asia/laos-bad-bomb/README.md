# Bad BOMB

| Field | Value |
|---|---|
| Country | Laos |
| Difficulty | Medium |
| Category | computer vision — unsupervised anomaly detection |
| Solved by | Stanislav |
| Status | solved |
| Flag | accepted on the platform (word not recorded here) |

## 60-second version

1225 unlabeled photos of the same green shell on two backgrounds; ~28 are defective (pass = 70% found).
Looking at the most unusual shells revealed three defect types: orange **scratch streaks**, a **notch** at the
seam (always with orange scratches), and small **holes** with an orange rim. Intact paint has no orange, so
counting orange pixels on the shell body separates 26 shells with a clear gap (≥ 25 orange pixels vs ≤ 8 for
normal shells). The last 2 places are filled by a PatchCore-style detector (ResNet18 patch features,
nearest-neighbour strangeness at the same position on all other shells).

## Task

Find the ~28 defective shells among the images; submit `test-output.txt` with their file names.
Score = |our list ∩ truth| / |truth|; ≥ 70% passes. No labels and no `train-output.txt` were provided,
so the format used is one file name per line (`image_0123.png`).

## Approach

```bash
python find_defective_shells.py   # -> test-output.txt, runs/all_scores.csv, runs/top_ranked_preview.jpg
```

1. Ignore 16 `image_000X(1).png` files (byte-identical copies) → 1225 unique images.
2. **Patch strangeness** (finds where shells differ): crop each shell to the same 448×160 frame (aligned on
   the green body), normalise lighting, ResNet18 layer2+layer3 features on an 8-px grid (56×20), and for each
   shell-mask cell the mean distance to its 20 nearest patches at the same/neighbouring position in other images.
3. **Orange count** (decides): pixels with hue 5–28, saturation ≥ 90, brightness ≥ 70 inside the shell body.
4. Ranking: orange count ≥ 20 first (by count), then the rest by patch strangeness → top 28.

| Group | Orange pixels | Shells |
|---|---|---|
| scratch streaks | 296–385 | 5 |
| notch + scratches | 101–150 | 10 |
| holes | 25–31 | 11 |
| normal | 0–8 | everything else |

RAM ≤ 1.6 GB; ~4 min first run (features cached in `runs/`), ~30 s afterwards.

## What failed

| # | Idea | Result | Lesson |
|---|---|---|---|
| 1 | patch kNN, k = 5, whole crop | ~16/28 of top look defective | defects repeat (~10 per type) → a defect finds similar defects as neighbours |
| 2 | k = 20 + score only shell cells | ~18/28, holes rank 22–44 | holes are smaller than a patch → diluted |
| 3 | orange-pixel count on the shell | 26 shells, clear gap | a domain-specific cue beats a generic model when the defect has a distinctive colour |

## Files

- `find_defective_shells.py` — full pipeline (patch strangeness + colour check), writes `test-output.txt`.
- `test-output.txt` — the submitted list.
- `task.md` — original task text. Data in `data/` (gitignored), intermediate files in `runs/` (gitignored).
