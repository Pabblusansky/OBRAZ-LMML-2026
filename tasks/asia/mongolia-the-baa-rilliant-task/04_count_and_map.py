"""Learn 'cell description -> number of sheep in this cell', check it on train, count the survey, draw the census map.

Data flow (shapes):
  train_cell_features.npy  (280, 38, 38, 384)   from 03_extract_cell_features.py
  train_points.csv         one (x, y) per sheep -> binned into a (280, 38, 38) 'sheep per cell' target
  ridge regression         384 numbers -> 1 number, fitted on 280*1444 = 404,320 cells
  predicted count          sum of the 1444 cell predictions of a photo
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = TASK_FOLDER / "data" / "task_mongolia"
RUNS_FOLDER = TASK_FOLDER / "runs"
sys.path.insert(0, str(TASK_FOLDER))
from importlib import import_module
survey_grid_positions = import_module("02_quick_map").survey_grid_positions

GRID_SIZE, IMAGE_SIZE, RIDGE_STRENGTH = 38, 600, 10.0


def sheep_per_cell(image_names, train_points):
    """(num_images, 38, 38): how many labelled sheep centres fall into each 16x16 cell, slightly blurred.
    Example: a dot at (x=250, y=100) -> cell column 250*38/600 = 15, cell row 100*38/600 = 6."""
    density_maps = np.zeros((len(image_names), GRID_SIZE, GRID_SIZE), np.float32)
    points_by_image = train_points.groupby("image")
    for image_index, image_name in enumerate(image_names):
        if image_name not in points_by_image.groups:
            continue
        dots = points_by_image.get_group(image_name)
        cell_columns = np.clip((dots.x * GRID_SIZE / IMAGE_SIZE).astype(int), 0, GRID_SIZE - 1)
        cell_rows = np.clip((dots.y * GRID_SIZE / IMAGE_SIZE).astype(int), 0, GRID_SIZE - 1)
        np.add.at(density_maps[image_index], (cell_rows, cell_columns), 1.0)
        # blur: a sheep straddling two cells gives each a share; the sum (= the count) stays the same
        density_maps[image_index] = gaussian_filter(density_maps[image_index], sigma=1.0, mode="constant")
    return density_maps


def fit_ridge(cell_features, cell_targets):
    """Closed-form ridge regression: weights = (X^T X + lambda*I)^-1 X^T y, with a bias column. X: (cells, 384)."""
    feature_mean = cell_features.mean(0)
    centred = cell_features - feature_mean
    gram_matrix = centred.T @ centred + RIDGE_STRENGTH * np.eye(centred.shape[1], dtype=np.float32)
    weights = np.linalg.solve(gram_matrix, centred.T @ (cell_targets - cell_targets.mean()))
    return weights, feature_mean, cell_targets.mean()


def predict_counts(model, image_features):
    """image_features: (num_images, 38, 38, 384) -> counts (num_images,)"""
    weights, feature_mean, target_mean = model
    flat = image_features.reshape(-1, image_features.shape[-1]).astype(np.float32)
    cell_predictions = (flat - feature_mean) @ weights + target_mean
    return cell_predictions.reshape(len(image_features), -1).sum(1)


if __name__ == "__main__":
    train_counts = pd.read_csv(DATA_FOLDER / "train_counts.csv")
    train_points = pd.read_csv(DATA_FOLDER / "train_points.csv")
    train_features = np.load(RUNS_FOLDER / "train_cell_features.npy")
    train_density = sheep_per_cell(train_counts.image, train_points)
    true_counts = train_counts["count"].to_numpy()

    # 5-fold cross-validation by photo: fit on 224 photos, count the other 56 -> honest error estimate
    fold_of_image = np.arange(len(true_counts)) % 5
    validation_counts = np.zeros(len(true_counts))
    for fold in range(5):
        is_train = fold_of_image != fold
        model = fit_ridge(train_features[is_train].reshape(-1, 384).astype(np.float32), train_density[is_train].ravel())
        validation_counts[~is_train] = predict_counts(model, train_features[~is_train])
    absolute_errors = np.abs(validation_counts - true_counts)
    print(f"cross-validated mean absolute error: {absolute_errors.mean():.1f} sheep "
          f"(true counts average {true_counts.mean():.1f}); correlation {np.corrcoef(validation_counts, true_counts)[0, 1]:.3f}")
    for name, truth, guess in list(zip(train_counts.image, true_counts, validation_counts))[:8]:
        print(f"  {name}: true {truth:3d}  predicted {guess:6.1f}")

    final_model = fit_ridge(train_features.reshape(-1, 384).astype(np.float32), train_density.ravel())
    survey_plots = pd.read_csv(DATA_FOLDER / "survey_plots.csv")
    survey_plots["count"] = np.clip(np.round(predict_counts(final_model, np.load(RUNS_FOLDER / "survey_cell_features.npy"))), 0, None).astype(int)
    survey_plots[["image", "count"]].to_csv(TASK_FOLDER / "counts.csv", index=False)
    survey_plots["row"], survey_plots["column"] = survey_grid_positions(survey_plots)

    count_grid = np.zeros((9, 37))
    count_grid[survey_plots.row, survey_plots.column] = survey_plots["count"]
    np.savetxt(RUNS_FOLDER / "count_grid.txt", count_grid, fmt="%4d")
    print(np.array2string(count_grid.astype(int), max_line_width=250))
    plt.figure(figsize=(18, 5))
    plt.imshow(count_grid, cmap="gray")
    plt.colorbar()
    plt.title("predicted sheep per plot (north up)")
    plt.savefig(RUNS_FOLDER / "census_map.png", dpi=70)
