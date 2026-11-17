"""
check_f1.py — score main.py on the training set the same way the hidden evaluator scores the test set.
Development only (not submitted).

USAGE (from this folder)
    python check_f1.py            # F1 with the settings in main.py, with and without group completion
    python check_f1.py --sweep    # also try other threshold combinations (measurements are computed only once)

HOW THE SCORE IS COMPUTED (example)
    true pairs (train-output.csv): 1399      our pairs: 1460      correct (in both): 1345
    precision = 1345 / 1460 = 0.921   recall = 1345 / 1399 = 0.961   F1 = 2PR / (P+R) = 0.941
    Pairs are unordered: (a, b) == (b, a), so both sides are stored sorted.
"""

import csv
import itertools
import sys
import time
from pathlib import Path

TASK_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_FOLDER))  # so `import main` works even with python -I
import main  # noqa: E402

TRAIN_IMAGES_FOLDER = TASK_FOLDER / "data" / "task_india" / "train"
TRAIN_TRUE_PAIRS_FILE = TASK_FOLDER / "data" / "task_india" / "train-output.csv"
PASS_THRESHOLD_F1 = 0.88


def read_pairs(csv_path):
    with open(csv_path) as csv_file:
        return {tuple(sorted((row["image_a"], row["image_b"]))) for row in csv.DictReader(csv_file)}


def score(predicted_pairs, true_pairs):
    predicted_pairs = set(predicted_pairs)
    correct_count = len(predicted_pairs & true_pairs)
    precision = correct_count / len(predicted_pairs) if predicted_pairs else 0.0
    recall = correct_count / len(true_pairs) if true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def print_score(label, predicted_pairs, true_pairs):
    precision, recall, f1 = score(predicted_pairs, true_pairs)
    verdict = "PASS" if f1 >= PASS_THRESHOLD_F1 else "FAIL"
    print(f"{label:38} pairs={len(set(predicted_pairs)):5}  precision={precision:.4f}  recall={recall:.4f}  F1={f1:.4f}  {verdict}")


def sweep_thresholds(prepared_images, candidate_pairs, measurements, true_pairs):
    """Try a grid of thresholds; print the best combinations. Stable neighbours = less risk of overfitting."""
    results = []
    for minimum_inliers, maximum_error, minimum_detail in itertools.product(
            [20, 40, 60, 100], [0.8, 1.3, 2.0, float("inf")], [0.4, 0.5, 0.6, 0.7]):
        main.MINIMUM_INLIERS, main.MAXIMUM_ALIGNMENT_ERROR_PX, main.MINIMUM_DETAIL_CORRELATION = minimum_inliers, maximum_error, minimum_detail
        pairs = main.select_duplicate_pairs(prepared_images, candidate_pairs, measurements)
        results.append((score(pairs, true_pairs)[2], minimum_inliers, maximum_error, minimum_detail))
    print("\nbest threshold combinations (F1, inliers>=, error<=, detail>=):")
    for result in sorted(results, reverse=True)[:8]:
        print(f"  F1={result[0]:.4f}  inliers>={result[1]:<4} error<={result[2]:<4} detail>={result[3]}")


def main_check():
    true_pairs = read_pairs(TRAIN_TRUE_PAIRS_FILE)
    image_paths = main.find_input_images(TRAIN_IMAGES_FOLDER)
    print(f"{len(image_paths)} training images, {len(true_pairs)} true pairs")

    start_time = time.time()
    prepared_images, candidate_pairs, measurements = main.prepare_and_measure(image_paths)
    print(f"preparation + verification took {time.time() - start_time:.0f} s")

    # How many true pairs survive the candidate step at all? (upper limit for recall)
    names = [image.name for image in prepared_images]
    candidate_name_pairs = {tuple(sorted((names[a], names[b]))) for a, b in candidate_pairs}
    print(f"true pairs among candidates: {len(true_pairs & candidate_name_pairs)}/{len(true_pairs)}\n")

    print_score("settings in main.py, no group completion",
                main.select_duplicate_pairs(prepared_images, candidate_pairs, measurements, use_group_completion=False), true_pairs)
    print_score("settings in main.py, group completion",
                main.select_duplicate_pairs(prepared_images, candidate_pairs, measurements, use_group_completion=True), true_pairs)

    if "--sweep" in sys.argv:
        sweep_thresholds(prepared_images, candidate_pairs, measurements, true_pairs)


if __name__ == "__main__":
    main_check()
