"""Step 4 - run the trained detector once and save EVERY detection (even weak ones) to CSV.

Why save weak detections too: the detector gives each box a confidence 0..1. Which cut-off
is right ("count boxes with confidence >= 0.4"?) is decided later in step 5 on the validation
images. Saving everything at a low cut-off (0.05) means we never have to re-run the slow
detector to try another threshold.

Output:
  runs/detections_val.csv     detections on the 270 held-out labelled rubbings
  runs/detections_crates.csv  detections on the 1,200 crate rubbings
  columns: image_path, class_index, class_name, confidence, x, y, w, h   (box in pixels)

    python 04_detect_glyphs.py
    python 04_detect_glyphs.py --weights runs/detector/weights/last.pt   # any other checkpoint
"""
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

TASK_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = TASK_FOLDER / "data" / "task_china"
VALIDATION_IMAGES_FOLDER = TASK_FOLDER / "data" / "yolo" / "images" / "val"
RUNS_FOLDER = TASK_FOLDER / "runs"
DETECTOR_WEIGHTS = RUNS_FOLDER / "detector" / "weights" / "best.pt"

IMAGE_SIZE = 640            # same size as in training
LOWEST_CONFIDENCE = 0.05    # keep almost everything; the real threshold is chosen in step 5


def detect_all(detector, image_paths):
    """Run the detector on a list of images; return one row per detected box."""
    detection_rows = []
    # One image per call on purpose: giving YOLO the whole list of 1,200 paths at once made
    # it load them together and peak at 8.8 GB RAM (the process was killed). One by one: ~1 GB.
    for image_number, image_path in enumerate(image_paths, start=1):
        result = detector.predict(str(image_path), imgsz=IMAGE_SIZE, conf=LOWEST_CONFIDENCE,
                                  device="cpu", verbose=False, max_det=300)[0]
        if image_number % 200 == 0:
            print(f"  {image_number}/{len(image_paths)} images done")
        boxes_xywh = result.boxes.xywh.numpy()    # (number_of_boxes, 4): center_x, center_y, w, h in pixels
        class_indices = result.boxes.cls.numpy().astype(int)
        confidences = result.boxes.conf.numpy()
        for (center_x, center_y, width, height), class_index, confidence in zip(boxes_xywh, class_indices, confidences):
            detection_rows.append({
                "image_path": str(image_path.relative_to(TASK_FOLDER)),
                "class_index": class_index,
                "class_name": result.names[class_index],
                "confidence": round(float(confidence), 4),
                "x": round(float(center_x - width / 2), 1),   # convert center -> top-left like train_labels.csv
                "y": round(float(center_y - height / 2), 1),
                "w": round(float(width), 1),
                "h": round(float(height), 1),
            })
    return pd.DataFrame(detection_rows)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--weights", type=Path, default=DETECTOR_WEIGHTS)
    arguments = argument_parser.parse_args()
    print("detector weights:", arguments.weights)
    detector = YOLO(str(arguments.weights))

    validation_image_paths = sorted(VALIDATION_IMAGES_FOLDER.glob("*.jpg"))
    validation_detections = detect_all(detector, validation_image_paths)
    validation_detections.to_csv(RUNS_FOLDER / "detections_val.csv", index=False)
    print(f"validation: {len(validation_image_paths)} images -> {len(validation_detections)} detections")

    crate_image_paths = sorted((DATASET_FOLDER / "crates").glob("crate_*/set_*/*.jpg"))
    crate_detections = detect_all(detector, crate_image_paths)
    crate_detections.to_csv(RUNS_FOLDER / "detections_crates.csv", index=False)
    print(f"crates: {len(crate_image_paths)} images -> {len(crate_detections)} detections")


if __name__ == "__main__":
    main()
