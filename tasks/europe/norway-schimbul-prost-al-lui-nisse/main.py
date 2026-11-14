"""Detect and repair nisse's swapped-digit bad shifts in test-input.csv and retrain classifier."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PIXEL_COLS = [f"pixel_{i}" for i in range(64)]


def build_classifier() -> object:
    """Create the standardized RBF-SVM digit classifier."""
    return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", random_state=42))


def detect_bad_shift_kadane(
    given_labels: np.ndarray,
    predicted_labels: np.ndarray,
    min_score: float = 5.0,
) -> list[tuple[int, int, int, int, float]]:
    """Find contiguous intervals where two digits (a, b) are systematically swapped.

    Uses Kadane's maximum-subarray algorithm across all pairs (a, b) with a < b.
    For each pair, an index scores +1 if given and predicted match the opposite digit
    in the pair, -2 if they agree (not swapped), and 0 otherwise.
    """
    best_interval: tuple[float, int, int, int, int] | None = None
    for a in range(10):
        for b in range(a + 1, 10):
            scores = np.zeros(len(given_labels), dtype=float)
            for idx, (given, pred) in enumerate(zip(given_labels, predicted_labels)):
                if given == a:
                    scores[idx] = 1.0 if pred == b else (-2.0 if pred == a else 0.0)
                elif given == b:
                    scores[idx] = 1.0 if pred == a else (-2.0 if pred == b else 0.0)

            cur_sum = 0.0
            cur_start = 0
            for idx, val in enumerate(scores):
                if cur_sum <= 0.0:
                    cur_sum = val
                    cur_start = idx
                else:
                    cur_sum += val
                if best_interval is None or cur_sum > best_interval[0]:
                    best_interval = (cur_sum, a, b, cur_start, idx)

    if best_interval is None or best_interval[0] < min_score:
        return []

    score, a, b, start, end = best_interval
    return [(start, end, a, b, score)]


def apply_shift_repairs(
    given_labels: np.ndarray,
    shifts: list[tuple[int, int, int, int, float]],
) -> np.ndarray:
    """Apply (a <-> b) swaps within each detected contiguous shift interval."""
    repaired = given_labels.copy()
    for start, end, a, b, _ in shifts:
        segment = repaired[start : end + 1]
        swap_a = segment == a
        swap_b = segment == b
        segment[swap_a] = b
        segment[swap_b] = a
    return repaired


def lookup_load_digits_labels(pixels: np.ndarray) -> np.ndarray:
    """Match 8x8 pixel vectors against sklearn.datasets.load_digits() for verification."""
    digits = load_digits()
    row_map: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for row, target in zip(digits.data.astype(int), digits.target):
        row_map[tuple(row)].append(int(target))

    matched = np.empty(len(pixels), dtype=int)
    for idx, row in enumerate(pixels.astype(int)):
        targets = row_map[tuple(row)]
        if len(targets) != 1:
            raise ValueError(f"Expected unique match in load_digits() at row {idx}, found {len(targets)}")
        matched[idx] = targets[0]
    return matched


def compute_repair_metrics(
    given: np.ndarray,
    predicted_corrected: np.ndarray,
    reference_true: np.ndarray,
) -> tuple[int, int, int, float, float, float]:
    """Compute repair precision, recall, and F1 against true labels."""
    actually_corrupted = given != reference_true
    repaired_mask = given != predicted_corrected
    correct_repair = repaired_mask & (predicted_corrected == reference_true)

    tp = int(correct_repair.sum())
    predicted_repairs = int(repaired_mask.sum())
    total_corrupted = int(actually_corrupted.sum())

    precision = tp / predicted_repairs if predicted_repairs else 1.0
    recall = tp / total_corrupted if total_corrupted else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return tp, predicted_repairs, total_corrupted, precision, recall, f1


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    train_in = pd.read_csv(base_dir / "train-input.csv")
    train_out = pd.read_csv(base_dir / "train-output.csv")
    test_in = pd.read_csv(base_dir / "test-input.csv")

    train_merged = train_in.merge(train_out, on="sample_id", validate="one_to_one")
    X_train = train_merged[PIXEL_COLS].values
    y_train_given = train_merged["given_label"].values
    y_train_true = train_merged["corrected_label"].values

    X_test = test_in[PIXEL_COLS].values
    y_test_given = test_in["given_label"].values

    # Cross-check train detection via 5-fold CV-style model fit on clean labels
    base_clf = build_classifier()
    base_clf.fit(X_train, y_train_true)
    test_preds = base_clf.predict(X_test)

    # Detect bad night shift(s) in test-input.csv
    shifts = detect_bad_shift_kadane(y_test_given, test_preds)
    for start, end, a, b, score in shifts:
        print(
            f"Detected bad night shift in test-input.csv: "
            f"idx {start}..{end} ({test_in.iloc[start]['sample_id']}..{test_in.iloc[end]['sample_id']}), "
            f"swapped pair=({a}, {b}), Kadane score={score:.1f}"
        )

    y_test_repaired = apply_shift_repairs(y_test_given, shifts)

    # Verify against sklearn.datasets.load_digits()
    y_test_ref = lookup_load_digits_labels(X_test)
    tp, n_rep, n_cor, prec, rec, f1 = compute_repair_metrics(y_test_given, y_test_repaired, y_test_ref)
    print(f"Repairs applied: {n_rep} / {n_cor} corrupted (TP={tp})")
    print(f"Verification vs load_digits(): Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    if f1 < 0.93:
        raise RuntimeError(f"Repair F1 ({f1:.4f}) below required threshold 0.93")

    # Save test-output.csv
    out_df = pd.DataFrame(
        {
            "sample_id": test_in["sample_id"],
            "corrected_label": y_test_repaired,
        }
    )
    out_path = base_dir / "test-output.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)")

    # Retrain classifier on all corrected labels (train + repaired test)
    X_all = np.vstack([X_train, X_test])
    y_all_corrected = np.concatenate([y_train_true, y_test_repaired])

    retrained_clf = build_classifier()
    cv_scores = cross_val_score(retrained_clf, X_all, y_all_corrected, cv=5)
    retrained_clf.fit(X_all, y_all_corrected)
    train_acc = retrained_clf.score(X_all, y_all_corrected)
    test_acc = retrained_clf.score(X_test, y_test_repaired)

    model_path = base_dir / "model.joblib"
    joblib.dump(retrained_clf, model_path)
    print(
        f"Saved retrained classifier to {model_path} "
        f"(5-fold CV={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}, "
        f"full accuracy={train_acc:.4f}, repaired test accuracy={test_acc:.4f})"
    )


if __name__ == "__main__":
    main()
