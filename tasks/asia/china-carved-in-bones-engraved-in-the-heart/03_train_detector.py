"""Step 3 - fine-tune a small pretrained YOLO detector on the labelled rubbings (CPU).

Why YOLO11n ("nano"): it already knows generic shapes/edges from millions of photos (COCO);
we only re-teach its last layers our 9 classes. Nano = 2.6 M parameters -> trainable on a CPU.

Input : data/yolo/dataset.yaml (made by 02_make_yolo_dataset.py)
Output: runs/detector/weights/best.pt  (+ training curves in runs/detector/)

    python 03_train_detector.py               # default 20 epochs (~5 min each on 12 CPU cores)
    python 03_train_detector.py --epochs 1    # quick timing test
    python 03_train_detector.py --resume      # continue after a crash (e.g. out of RAM)
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

TASK_FOLDER = Path(__file__).resolve().parent
DATASET_YAML = TASK_FOLDER / "data" / "yolo" / "dataset.yaml"
RUNS_FOLDER = TASK_FOLDER / "runs"
PRETRAINED_WEIGHTS = Path("/home/cs/LMML/models/yolo11n.pt")   # COCO-pretrained nano model (outside the repo)

IMAGE_SIZE = 640     # rubbings are ~600-800 px; a median glyph (~60x95 px) stays ~50-80 px after resizing
BATCH_SIZE = 8       # images per training step; 16 used 4.3 GB RAM on CPU, 8 keeps it lower
DATA_WORKERS = 4     # parallel image-loading processes (each one costs RAM)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--epochs", type=int, default=20)
    argument_parser.add_argument("--name", default="detector")
    argument_parser.add_argument("--resume", action="store_true",
                                 help="continue an interrupted run from runs/<name>/weights/last.pt")
    arguments = argument_parser.parse_args()

    if arguments.resume:
        # last.pt stores the weights AND the training state (epoch number, optimizer, settings),
        # so training continues e.g. at epoch 4 of 20 instead of starting over.
        interrupted_checkpoint = RUNS_FOLDER / arguments.name / "weights" / "last.pt"
        YOLO(str(interrupted_checkpoint)).train(resume=True)
        return

    detector = YOLO(str(PRETRAINED_WEIGHTS))
    detector.train(
        data=str(DATASET_YAML),
        epochs=arguments.epochs,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=DATA_WORKERS,
        device="cpu",
        project=str(RUNS_FOLDER),
        name=arguments.name,
        exist_ok=True,
        seed=0,
        # Augmentation notes:
        #  fliplr=0.5 (default) is fine: oracle-bone scribes often wrote a character mirrored,
        #  so a left-right flipped glyph is still the same glyph.
        #  flipud stays 0 (default): characters are never written upside down.
        fliplr=0.5,
        flipud=0.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
