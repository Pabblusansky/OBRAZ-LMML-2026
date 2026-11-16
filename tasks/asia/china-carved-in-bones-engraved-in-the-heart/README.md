# Carved in bones, engraved in the heart

| Field | Value |
|---|---|
| Country | China |
| Difficulty | Medium |
| Category | computer vision — object detection + counting, then a 5-bit decode |
| Solved by | |
| Status | flag decoded — awaiting platform result |
| Flag | `SIGMOID_FENGSHUI` (fill in after the platform accepts it) |

## 60-second version

1,200 unlabelled rubbings of Shang oracle bones sit in 8 crates × 5 sets × 30 images. Each
crate has one "lid glyph". We fine-tuned a small YOLO detector on the 1,800 labelled rubbings
(8 crate glyphs + one `other_glyph` class for the remaining 203 glyph ids), counted each crate's
glyph in each of its 5 sets, and got counts that are clearly either **many (≈10–20)** or
**few (≈0–3)**. Many = 1, few = 0, set_1 = highest bit → a 5-bit number 1–26 → a letter.
The 8 crates read **F E N G S H U I** → `SIGMOID_FENGSHUI`.

## Task

Storyline: an archivist in Anyang (Yinxu, pit YH127) sorted oracle-bone rubbings into 8 crates
with one glyph painted on each lid, 5 bundles per crate. "The bones answer whoever counts
carefully enough." Objective: each crate keeps one letter, written in how often the crate's glyph
was carved across its five sets; the 8 letters form the word. Deliverables: the flag, the code,
`counts.csv` (`crate, set, glyph_id, count`, one row per set) and a short write-up.

## Approach

```bash
# dataset unpacked to data/task_china/ (train/, crates/, *.csv)
python 01_explore.py            # look at the data: boxes, crate samples, glyph crops -> runs/
python 02_make_yolo_dataset.py  # train_labels.csv -> YOLO labels, 85/15 train/val split by image
python 03_train_detector.py     # fine-tune YOLO11n (COCO-pretrained), 20 epochs, CPU
python 04_detect_glyphs.py      # all detections (conf >= 0.05) on val + crate images -> runs/*.csv
python 05_choose_threshold.py   # per-glyph confidence threshold that minimises count error on val
python 06_count_and_decode.py   # counts.csv + decoded word
```

1. **Data.** Greyscale JPEGs of different sizes (≈250–800 px), white glyphs on black ink;
   ~8 labelled glyphs per image, 211 glyph ids. Crate glyphs have 137 (今) to 1,089 (貞)
   training boxes. No crate image is a near-copy of a training image (best thumbnail
   correlation 0.91), so real detection is needed.
2. **Detector.** YOLO11n, 9 classes: the 8 crate glyphs + `other_glyph` (all other ids incl.
   `rare`). Showing the detector the other glyphs as their own class teaches it what *not* to
   count. Left–right flips are kept as augmentation (oracle-bone characters are often written
   mirrored); no up–down flips. imgsz 640, batch 8 (~3.6 GB RAM), ~5 min/epoch on 12 CPU cores.
3. **Counting.** A box counts if its class is the crate's glyph and its confidence ≥ a
   per-glyph threshold chosen on 270 held-out labelled rubbings (minimum total |predicted − true|
   per image). Only the crate's own glyph is counted in each crate.
4. **Decoding.** Per crate, sort the 5 counts and split them at the biggest jump into
   few (0) / many (1). Bits in set order, set_1 = most significant; value 1–26 → A–Z.
   No hand-picked cut-off is needed: every letter A (00001) … Z (11010) has at least one 0 and
   at least one 1 bit, so every crate truly contains both groups.

| crate | glyph | counts set_1..5 | bits | value | letter |
|---|---|---|---|---|---|
| crate_01 | g015 賓 | 2, 1, 15, 14, 2 | 00110 | 6 | F |
| crate_02 | g028 今 | 2, 0, 14, 0, 13 | 00101 | 5 | E |
| crate_03 | g005 其 | 0, 13, 14, 14, 2 | 01110 | 14 | N |
| crate_04 | g018 㞢 | 1, 1, 15, 13, 17 | 00111 | 7 | G |
| crate_05 | g003 王 | 17, 1, 0, 14, 12 | 10011 | 19 | S |
| crate_06 | g011 不 | 1, 7, 1, 1, 1 | 01000 | 8 | H |
| crate_07 | g001 貞 | 14, 3, 14, 1, 15 | 10101 | 21 | U |
| crate_08 | g010 雨 | 3, 10, 2, 1, 17 | 01001 | 9 | I |

Final model: 20 epochs (1.7 h on CPU), validation mAP50 0.84, mAP50-95 0.46. Counting accuracy on the
270 held-out rubbings: 92–98 % of images have exactly the right count per crate glyph; the error of a
30-image sum is 0.5–1.6 on average — far below the gap between "few" (≤ 3) and "many" (≥ 7).
An early checkpoint (epoch 6, mAP50 0.72) already decoded the same word; the final model makes the
gaps wider (smallest: crate_06, 1 vs 7).

## Why it worked

- **The encoding tolerates counting errors.** A letter depends only on whether a set has
  *many* or *few* of the glyph, not on the exact number, so a detector that misses or adds one
  or two glyphs per set still gives the right bits. Even an early checkpoint decoded the word.
- **A background class.** Other glyphs (some look like crate glyphs) are labelled
  `other_glyph` instead of being ignored, so the detector learns to reject them.
- **Threshold chosen on held-out data,** per glyph, by the quantity we actually need (count
  error), not by a generic detection metric.

## What failed / pitfalls

| # | Attempt | Result | Lesson |
|---|---|---|---|
| 1 | batch 16 for training | 4.3 GB RAM, machine nearly out of memory | batch 8 on this laptop |
| 2 | a second detection job next to training | out of memory, PC restarted | one heavy job at a time; `--resume` continues from `last.pt` |
| 3 | `predict()` on a list of 1,200 paths | peaked at 8.8 GB, killed | predict one image per call (0.76 GB peak) |
| 4 | crate_06 (不) with the epoch-6 model | "many" set counted only 4 (val: 21 of 36 found) | weakest glyph; final model: 7 vs ≤ 1 |

## Files

| File | What |
|---|---|
| `01_explore.py` … `06_count_and_decode.py` | the pipeline, run in order (commands above) |
| `counts.csv` | crate, set, glyph_id, count — 40 rows |
| `requirements.txt` | Python packages |
| `task.md` | original task text |
| `data/`, `runs/` | dataset, YOLO labels, weights, detections (gitignored) |
