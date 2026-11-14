"""
check_reconstruction.py — score main.py locally like the evaluator does. Development only (not submitted).

USAGE (from this folder)
    python check_reconstruction.py            # train image: order accuracy, SSIM, time
    python check_reconstruction.py --stress   # + noisy strips and other images cut into strips

SCORES
    order accuracy   = fraction of positions holding the right strip; the mirrored order also counts
                       e.g. truth [a,b,c,d], ours [a,b,d,c] -> 2/4 = 0.50   (needs >= 0.90)
    neighbour accuracy = fraction of true neighbour pairs that are neighbours in our order too
                       (more forgiving: one misplaced block only breaks 2-3 joins)
    SSIM             = structural similarity between our image and the true image (needs >= 0.90)
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

TASK_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_FOLDER))
import main  # noqa: E402

TRAIN_INPUT = TASK_FOLDER / "data" / "train-input"
TRAIN_ORDER = TASK_FOLDER / "data" / "train-output" / "reconstructed_order.json"
TRAIN_IMAGE = TASK_FOLDER / "data" / "train-output" / "reconstructed_image.png"
OTHER_IMAGES = sorted((TASK_FOLDER.parent / "india-twins-separated-at-birth" / "data" / "task_india" / "train").glob("*.jpg"))


def order_accuracy(predicted_order, true_order):
    forward = np.mean([a == b for a, b in zip(predicted_order, true_order)])
    mirrored = np.mean([a == b for a, b in zip(predicted_order[::-1], true_order)])
    return max(forward, mirrored)


def neighbour_accuracy(predicted_order, true_order):
    true_joins = set(zip(true_order, true_order[1:]))
    predicted_joins = set(zip(predicted_order, predicted_order[1:]))
    mirrored_joins = {(b, a) for a, b in predicted_joins}
    return max(len(true_joins & predicted_joins), len(true_joins & mirrored_joins)) / len(true_joins)


def ssim(image_a, image_b):
    return structural_similarity(image_a, image_b, channel_axis=2, data_range=255)


def solve_in_memory(strips):
    """Same steps as main.py, on a list of strip arrays (no files) -> order as list of indices."""
    colour_strips = np.stack([strip.astype(np.float32) for strip in strips])
    return main.order_strips(main.normalise_join_costs(main.compute_join_costs(colour_strips)))


def check_train_image():
    start_time = time.time()
    strip_names, original_strips, order = main.reconstruct(TRAIN_INPUT)
    elapsed = time.time() - start_time
    predicted_names = [strip_names[index] for index in order]
    true_names = json.load(open(TRAIN_ORDER))
    our_image = np.hstack([original_strips[index] for index in order])
    true_image = cv2.imread(str(TRAIN_IMAGE))
    if order_accuracy(predicted_names[::-1], true_names) > order_accuracy(predicted_names, true_names):
        our_image = np.hstack([original_strips[index] for index in order[::-1]])  # compare the mirror fairly
    print(f"TRAIN  strips={len(strip_names)}  order={order_accuracy(predicted_names, true_names):.3f}  "
          f"neighbours={neighbour_accuracy(predicted_names, true_names):.3f}  SSIM={ssim(our_image, true_image):.3f}  "
          f"time={elapsed:.2f}s")


def shred(image, strip_width, random_generator, noise_level=0.0):
    """Cut an image into vertical strips, shuffle them, optionally add noise. Returns strips + true order."""
    usable_width = image.shape[1] // strip_width * strip_width
    strips = [image[:, x:x + strip_width] for x in range(0, usable_width, strip_width)]
    if noise_level:
        strips = [np.clip(strip + random_generator.normal(0, noise_level, strip.shape), 0, 255).astype(np.uint8) for strip in strips]
    shuffled_positions = random_generator.permutation(len(strips))
    return [strips[position] for position in shuffled_positions], list(np.argsort(shuffled_positions))


def stress_test():
    random_generator = np.random.default_rng(0)
    true_image = cv2.imread(str(TRAIN_IMAGE))

    print("\nnoise added to every strip (train image, 192 strips):")
    for noise_level in [5, 10, 20]:
        strips, true_order = shred(true_image, 8, random_generator, noise_level)
        order = solve_in_memory(strips)
        print(f"  noise sigma={noise_level:<3} order={order_accuracy(order, true_order):.3f}  neighbours={neighbour_accuracy(order, true_order):.3f}")

    print("\nother images (India photos resized to 1536x1024, 192 strips of 8 px):")
    scores = []
    for image_path in OTHER_IMAGES[::200][:15]:
        image = cv2.resize(cv2.imread(str(image_path)), (1536, 1024), interpolation=cv2.INTER_CUBIC)
        strips, true_order = shred(image, 8, random_generator)
        order = solve_in_memory(strips)
        scores.append(order_accuracy(order, true_order))
        print(f"  {image_path.name}  order={scores[-1]:.3f}  neighbours={neighbour_accuracy(order, true_order):.3f}")
    print(f"  -> passed (order >= 0.90): {sum(score >= 0.9 for score in scores)}/{len(scores)}")


if __name__ == "__main__":
    check_train_image()
    if "--stress" in sys.argv:
        stress_test()
