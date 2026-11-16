"""Step 2 - convert train_labels.csv into the folder layout a YOLO detector trains on.

Classes (9):
  0..7  the 8 crate glyphs, in crates.csv order (crate_01 -> class 0, ..., crate_08 -> class 7)
  8     other_glyph = every other glyph id (including "rare"). The detector still has to
        see them, so it learns "this is writing, but NOT a crate glyph -> don't count it".

YOLO label format: one .txt per image, one line per box:
    class_id  center_x  center_y  width  height        (all divided by the image size -> 0..1)
Example: image 400 px wide x 600 px high, box x=100 y=300 w=40 h=60, class 2 ->
    "2 0.300000 0.550000 0.100000 0.100000"
    center_x = (100 + 40/2) / 400 = 0.30,  center_y = (300 + 60/2) / 600 = 0.55

Output (gitignored): data/yolo/{images,labels}/{train,val}/  +  data/yolo/dataset.yaml
Images are symlinked, not copied (saves disk).

    python 02_make_yolo_dataset.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TASK_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = TASK_FOLDER / "data" / "task_china"
YOLO_FOLDER = TASK_FOLDER / "data" / "yolo"
VALIDATION_FRACTION = 0.15   # 15 % of the labelled rubbings are kept aside to measure counting accuracy
RANDOM_SEED = 0


def load_tables():
    train_labels = pd.read_csv(DATASET_FOLDER / "train_labels.csv", encoding="utf-8-sig")  # utf-8-sig: strip BOM
    crates = pd.read_csv(DATASET_FOLDER / "crates.csv", encoding="utf-8-sig")
    return train_labels, crates


def class_names_and_mapping(crates):
    """Return (class names list, dict glyph_id -> class index). Non-crate glyphs are missing from the dict."""
    crate_glyph_ids = list(crates.glyph_id)                       # ['g015', 'g028', ...]
    class_names = crate_glyph_ids + ["other_glyph"]
    glyph_id_to_class = {glyph_id: index for index, glyph_id in enumerate(crate_glyph_ids)}
    return class_names, glyph_id_to_class


def split_images(image_names):
    """Random split by IMAGE (not by box) so validation rubbings are fully unseen."""
    random_generator = np.random.default_rng(RANDOM_SEED)
    shuffled_names = random_generator.permutation(sorted(image_names))
    number_of_validation_images = int(len(shuffled_names) * VALIDATION_FRACTION)
    validation_names = set(shuffled_names[:number_of_validation_images])
    return {name: ("val" if name in validation_names else "train") for name in shuffled_names}


def box_to_yolo_line(box, class_index, image_width, image_height):
    # clip the box to the image (a few labelled boxes stick out by a pixel or two)
    left = max(0.0, box.x)
    top = max(0.0, box.y)
    right = min(float(image_width), box.x + box.w)
    bottom = min(float(image_height), box.y + box.h)
    center_x = (left + right) / 2 / image_width
    center_y = (top + bottom) / 2 / image_height
    width = (right - left) / image_width
    height = (bottom - top) / image_height
    return f"{class_index} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def main():
    train_labels, crates = load_tables()
    class_names, glyph_id_to_class = class_names_and_mapping(crates)
    other_glyph_class = len(class_names) - 1                      # = 8
    image_to_split = split_images(train_labels.image.unique())

    for split_name in ("train", "val"):
        (YOLO_FOLDER / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (YOLO_FOLDER / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for image_name, image_boxes in train_labels.groupby("image"):
        split_name = image_to_split[image_name]
        source_image_path = DATASET_FOLDER / "train" / "images" / image_name
        image_height, image_width = cv2.imread(str(source_image_path), cv2.IMREAD_GRAYSCALE).shape

        yolo_lines = [
            box_to_yolo_line(box, glyph_id_to_class.get(box.glyph_id, other_glyph_class), image_width, image_height)
            for box in image_boxes.itertuples()
        ]
        label_path = YOLO_FOLDER / "labels" / split_name / (Path(image_name).stem + ".txt")
        label_path.write_text("\n".join(yolo_lines) + "\n")

        linked_image_path = YOLO_FOLDER / "images" / split_name / image_name
        if not linked_image_path.exists():
            linked_image_path.symlink_to(source_image_path)

    dataset_yaml = (
        f"path: {YOLO_FOLDER}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(class_names))
    )
    (YOLO_FOLDER / "dataset.yaml").write_text(dataset_yaml)

    split_counts = pd.Series(image_to_split).value_counts().to_dict()
    print("images per split:", split_counts)
    print("classes:", class_names)
    print("dataset description written to", YOLO_FOLDER / "dataset.yaml")


if __name__ == "__main__":
    main()
