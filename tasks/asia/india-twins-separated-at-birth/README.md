# Twins Separated at Birth

| Field | Value |
|---|---|
| Country | India |
| Difficulty | Easy |
| Category | computer vision — near-duplicate image detection |
| Solved by | Stanislav |
| Status | solved |
| Flag | accepted on the platform (word not recorded here) |

## 60-second version

4,057 photos of Indian monuments; some are edited copies (resized, cropped, rotated a few
degrees, re-saved, brightened) of the same source photo. The trap: many *different* photos of
the same monument, often taken seconds apart from almost the same spot. `main.py` first finds
~10 look-alike candidates per image with a cheap fingerprint, then verifies each candidate by
matching ORB keypoints, fitting one rotation+scale+shift with RANSAC, aligning the two images and
comparing their fine details. Edited copies align perfectly; photos taken seconds apart do not
(clouds, leaves, people and parallax differ). Train F1 = 0.957 (precision 0.924, recall 0.993).

## Task

Find every unordered pair of images that originates from the same source photograph.
Submit `main.py`; it must read the input archive and write `output.csv` (`image_a,image_b`).
Scored by F1 against hidden pairs; F1 ≥ 0.88 reveals the flag.

## Approach

```bash
python main.py              # searches the working folder (images or .zip) -> output.csv
python main.py <folder|zip> # explicit input
python check_f1.py --sweep  # local score on train + threshold search (development only)
```

1. **Candidates** (cheap): each image → 16×16 grayscale thumbnail of its centre (256 numbers,
   mean-removed) + hue/saturation histogram (128 numbers). Similarity = thumbnail correlation +
   histogram similarity. Keep each image's 10 most similar → 32,728 candidate pairs instead of
   8.2 M possible pairs; 1398 of 1399 true pairs survive.
2. **Verification** (exact), per candidate pair:
   - ORB keypoints (1000 per image, 512 px copy) + ratio-test matching;
   - RANSAC fit of a similarity transform (`cv2.estimateAffinePartial2D`) → number of inliers;
   - median alignment error of the inliers (same photo: ~0.01–0.8 px);
   - warp image A onto B, correlate the **fine details** (image − Gaussian blur σ=3) in the overlap.
3. **Decision**: inliers ≥ 40 **and** alignment error ≤ 1.3 px **and** detail correlation ≥ 0.6.

Runtime ≈ 60 s for 4,057 images on 12 CPU cores, ~1.2 GB RAM. Needs `numpy` and `opencv-python`.

## Why it works

- Edits applied to a copy (resize, crop, small rotation, re-compression, brightness) are all
  explained by **one** rotation + scale + shift, so hundreds of keypoints agree on it.
- A second photo of the same scene was taken from a slightly different position: near and far
  objects shift differently (parallax), so no single transform fits all points; and moving
  things (clouds, birds, people, leaves) differ, so the aligned fine details do not correlate.
- Comparing fine details instead of raw pixels ignores the large shared structure (sky, building
  outline) that makes any two photos of the same monument look alike.

## What failed

| # | Idea | Result | Lesson |
|---|---|---|---|
| 1 | dHash (64-bit perceptual hash) for candidates | true twin in top-10 only 84% | rotation/crop shift the hash bits |
| 2 | thumbnail + colour histogram for candidates | true twin in top-10 **99.5%** | kept |
| 3 | verify with inliers + raw pixel correlation | F1 0.940 | raw correlation is high for any photo of the same monument |
| 4 | add alignment error + fine-detail correlation | F1 **0.957** | kept |
| 5 | complete groups (A~B, B~C ⇒ A~C) | F1 0.949 (worse) | one false match merges two groups into several false pairs |
| — | file numbers of pairs are always adjacent in train | **not used** | leakage: the test set is shuffled |

### Submission log

| # | Platform feedback | Cause / action |
|---|---|---|
| 1 | "Staging failed: task input missing: test-input.zip" | evaluator had no test file → reported to organisers |
| 2 | `zipfile.BadZipFile: File is not a zip file` | made input handling robust (detect archive type by content, no crash, diagnostics) |
| 3 | "no images found" + diagnostics: `test-input.zip: 134 bytes, starts with b'version https://'` | the evaluator's `test-input.zip` is a **Git LFS pointer**, not the archive (needs `git lfs pull` on their side) → reported to organisers |

## Files

- `main.py` — the submission (self-contained).
- `check_f1.py` — scores `main.py` on the training data (`data/task_india/`, gitignored).
- `task.md` — original task text.
