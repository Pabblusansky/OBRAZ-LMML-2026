"""Step 1 - look at the data before modelling anything.

Writes to runs/:
  explore_train_with_boxes.png  - 6 labelled rubbings, crate glyph boxes in colour
  explore_crate_samples.png     - 6 unlabelled rubbings from one crate set
  explore_crate_glyph_crops.png - 12 example crops of each of the 8 crate glyphs

    python 01_explore.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TASK_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = TASK_FOLDER / "data" / "task_china"
RUNS_FOLDER = TASK_FOLDER / "runs"
RUNS_FOLDER.mkdir(exist_ok=True)


def load_tables():
    # the CSVs start with a UTF-8 "BOM" byte marker; "utf-8-sig" strips it,
    # otherwise the first column would be called "﻿image" instead of "image"
    train_labels = pd.read_csv(DATASET_FOLDER / "train_labels.csv", encoding="utf-8-sig")
    crates = pd.read_csv(DATASET_FOLDER / "crates.csv", encoding="utf-8-sig")
    return train_labels, crates


def side_by_side(images, height=500):
    """Resize every image to the same height and glue them horizontally."""
    resized = [cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height)) for image in images]
    return np.hstack(resized)


def main():
    train_labels, crates = load_tables()
    crate_glyph_ids = list(crates.glyph_id)
    print(f"train: {train_labels.image.nunique()} images, {len(train_labels)} boxes, "
          f"{train_labels.glyph_id.nunique()} glyph ids")

    # How common are the 8 crate glyphs in training? (= how many examples we can learn from)
    for crate_name, glyph_id, character in crates.itertuples(index=False):
        number_of_boxes = (train_labels.glyph_id == glyph_id).sum()
        images_containing_it = train_labels[train_labels.glyph_id == glyph_id].image.nunique()
        print(f"  {crate_name}  {glyph_id} {character}: {number_of_boxes} boxes in {images_containing_it} images")

    # --- 6 training rubbings with their boxes: crate glyphs red, all other glyphs grey
    example_images = train_labels.image.drop_duplicates().iloc[:6]
    drawn_images = []
    for image_name in example_images:
        grayscale_image = cv2.imread(str(DATASET_FOLDER / "train" / "images" / image_name), cv2.IMREAD_GRAYSCALE)
        colour_image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)  # so we can draw coloured boxes
        for box in train_labels[train_labels.image == image_name].itertuples():
            is_crate_glyph = box.glyph_id in crate_glyph_ids
            box_colour = (0, 0, 255) if is_crate_glyph else (160, 160, 160)  # OpenCV colours are BGR
            top_left = (int(box.x), int(box.y))
            bottom_right = (int(box.x + box.w), int(box.y + box.h))
            cv2.rectangle(colour_image, top_left, bottom_right, box_colour, 2)
            cv2.putText(colour_image, box.glyph_id, (top_left[0], top_left[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colour, 1)
        drawn_images.append(colour_image)
    cv2.imwrite(str(RUNS_FOLDER / "explore_train_with_boxes.png"), side_by_side(drawn_images))

    # --- 6 unlabelled rubbings from crate_01/set_1
    crate_image_paths = sorted((DATASET_FOLDER / "crates" / "crate_01" / "set_1").glob("*.jpg"))[:6]
    crate_images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in crate_image_paths]
    cv2.imwrite(str(RUNS_FOLDER / "explore_crate_samples.png"), side_by_side(crate_images))

    # --- a grid: one row per crate glyph, 12 example crops each (64x64 px)
    crop_size = 64
    grid_rows = []
    for glyph_id in crate_glyph_ids:
        glyph_boxes = train_labels[train_labels.glyph_id == glyph_id].sample(12, random_state=0)
        crops = []
        for box in glyph_boxes.itertuples():
            grayscale_image = cv2.imread(str(DATASET_FOLDER / "train" / "images" / box.image), cv2.IMREAD_GRAYSCALE)
            crop = grayscale_image[int(box.y):int(box.y + box.h), int(box.x):int(box.x + box.w)]
            crops.append(cv2.resize(crop, (crop_size, crop_size)))
        grid_rows.append(np.hstack(crops))
    cv2.imwrite(str(RUNS_FOLDER / "explore_crate_glyph_crops.png"), np.vstack(grid_rows))
    print("images written to", RUNS_FOLDER)


if __name__ == "__main__":
    main()
