"""Step 5 - measure how well we COUNT on held-out labelled rubbings, and pick the confidence threshold.

The detector gives every box a confidence. Counting rule: count = number of boxes of the glyph
with confidence >= threshold.
  threshold too low  -> extra boxes on scratches / similar glyphs -> count too HIGH
  threshold too high -> real but faint glyphs are dropped          -> count too LOW
We try thresholds 0.05, 0.10, ..., 0.90 and, per crate glyph, keep the one with the smallest
counting error on the 270 validation rubbings (whose true counts we know from train_labels.csv).

Example for one glyph on 3 images: true counts [1, 0, 2]
  threshold 0.3 -> predicted [1, 1, 2] -> absolute errors [0, 1, 0] -> total error 1
  threshold 0.5 -> predicted [1, 0, 2] -> absolute errors [0, 0, 0] -> total error 0  <- chosen

Input : runs/detections_val.csv (step 4), train_labels.csv, crates.csv
Output: runs/thresholds.csv  (glyph_id, threshold, validation errors)

    python 05_choose_threshold.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

TASK_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = TASK_FOLDER / "data" / "task_china"
RUNS_FOLDER = TASK_FOLDER / "runs"
CANDIDATE_THRESHOLDS = np.round(np.arange(0.05, 0.91, 0.05), 2)
SET_SIZE = 30            # a crate set has 30 rubbings -> also check the error of a 30-image SUM
NUMBER_OF_FAKE_SETS = 200


def true_counts_per_image(validation_image_names, glyph_ids):
    """Table: rows = validation images, columns = crate glyph ids, values = true number of boxes."""
    train_labels = pd.read_csv(DATASET_FOLDER / "train_labels.csv", encoding="utf-8-sig")
    validation_labels = train_labels[train_labels.image.isin(validation_image_names)]
    counts = pd.crosstab(validation_labels.image, validation_labels.glyph_id)
    # reindex: images with none of a glyph get 0, glyphs we don't care about are dropped
    return counts.reindex(index=validation_image_names, columns=glyph_ids, fill_value=0)


def predicted_counts_per_image(detections, validation_image_names, glyph_id, threshold):
    confident_boxes = detections[(detections.class_name == glyph_id) & (detections.confidence >= threshold)]
    return confident_boxes.image_name.value_counts().reindex(validation_image_names, fill_value=0)


def fake_set_errors(true_counts, predicted_counts, random_generator):
    """Error of the SUM over random groups of 30 images - this is what a crate set count looks like."""
    errors = []
    for _ in range(NUMBER_OF_FAKE_SETS):
        chosen = random_generator.choice(len(true_counts), SET_SIZE, replace=False)
        errors.append(predicted_counts.values[chosen].sum() - true_counts.values[chosen].sum())
    return np.array(errors)


def main():
    crates = pd.read_csv(DATASET_FOLDER / "crates.csv", encoding="utf-8-sig")
    glyph_ids = list(crates.glyph_id)
    detections = pd.read_csv(RUNS_FOLDER / "detections_val.csv")
    detections["image_name"] = detections.image_path.map(lambda path: Path(path).name)
    validation_image_names = sorted(path.name for path in (TASK_FOLDER / "data" / "yolo" / "images" / "val").glob("*.jpg"))
    true_counts = true_counts_per_image(validation_image_names, glyph_ids)

    random_generator = np.random.default_rng(0)
    chosen_rows = []
    for glyph_id, character in zip(crates.glyph_id, crates.modern_character):
        true_glyph_counts = true_counts[glyph_id]
        best = None
        for threshold in CANDIDATE_THRESHOLDS:
            predicted = predicted_counts_per_image(detections, validation_image_names, glyph_id, threshold)
            total_absolute_error = int((predicted - true_glyph_counts).abs().sum())
            if best is None or total_absolute_error < best["total_absolute_error"]:
                best = {"threshold": threshold, "total_absolute_error": total_absolute_error, "predicted": predicted}

        predicted = best["predicted"]
        images_exactly_right = float((predicted == true_glyph_counts).mean())
        set_errors = fake_set_errors(true_glyph_counts, predicted, random_generator)
        chosen_rows.append({
            "glyph_id": glyph_id,
            "character": character,
            "threshold": best["threshold"],
            "true_total": int(true_glyph_counts.sum()),
            "predicted_total": int(predicted.sum()),
            "images_exactly_right": round(images_exactly_right, 3),
            "set_of_30_mean_abs_error": round(float(np.abs(set_errors).mean()), 2),
            "set_of_30_max_abs_error": int(np.abs(set_errors).max()),
        })

    summary = pd.DataFrame(chosen_rows)
    summary.to_csv(RUNS_FOLDER / "thresholds.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
