"""Quick look at the hidden word BEFORE building a good counter.

Idea: sheep from above are bright, almost colourless blobs on green/brown grass.
So 'fraction of bright low-saturation pixels' is a rough sheep score.
Example: a pixel HSV (hue 40, saturation 20, value 230) -> wool; (60, 150, 120) -> grass.
We check on train how well this score follows the true counts, then paint the survey grid with it.
"""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = TASK_FOLDER / "data" / "task_mongolia"
RUNS_FOLDER = TASK_FOLDER / "runs"


def wool_pixel_fraction(image_path):
    hsv_image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2HSV)
    saturation, brightness = hsv_image[..., 1], hsv_image[..., 2]
    wool_mask = (brightness > 170) & (saturation < 60)
    return wool_mask.mean()


def survey_grid_positions(survey_plots):
    """Rotate the tilted plot grid so it is axis-aligned, then give every plot a (row, column)."""
    centred_xy = survey_plots[["x_m", "y_m"]].to_numpy() - survey_plots[["x_m", "y_m"]].mean().to_numpy()
    # SVD/PCA: first direction = the long side of the field (about 27 degrees), second = the short side
    _, _, principal_directions = np.linalg.svd(centred_xy, full_matrices=False)
    if principal_directions[0, 0] < 0:          # make 'along' point to +x (west -> east)
        principal_directions[0] *= -1
    if principal_directions[1, 1] < 0:          # make 'across' point to +y (south -> north)
        principal_directions[1] *= -1
    along_m, across_m = (centred_xy @ principal_directions.T).T
    # plots are ~50 m apart with a few metres of jitter -> divide by the spacing and round
    column = np.round((along_m - along_m.min()) / ((along_m.max() - along_m.min()) / 36)).astype(int)
    row_from_south = np.round((across_m - across_m.min()) / ((across_m.max() - across_m.min()) / 8)).astype(int)
    return 8 - row_from_south, column           # row 0 = north (top of the picture)


if __name__ == "__main__":
    train_counts = pd.read_csv(DATA_FOLDER / "train_counts.csv")
    train_counts["wool_fraction"] = [wool_pixel_fraction(DATA_FOLDER / "train" / name) for name in train_counts.image]
    print("correlation(wool fraction, true count) =", round(train_counts[["wool_fraction", "count"]].corr().iloc[0, 1], 3))

    survey_plots = pd.read_csv(DATA_FOLDER / "survey_plots.csv")
    survey_plots["wool_fraction"] = [wool_pixel_fraction(DATA_FOLDER / "survey" / name) for name in survey_plots.image]
    survey_plots["row"], survey_plots["column"] = survey_grid_positions(survey_plots)
    print("cells filled:", survey_plots.groupby(["row", "column"]).ngroups, "of 333")

    score_grid = np.full((9, 37), np.nan)
    score_grid[survey_plots.row, survey_plots.column] = survey_plots.wool_fraction
    plt.figure(figsize=(18, 5))
    plt.imshow(score_grid, cmap="gray")
    plt.colorbar()
    plt.title("wool pixel fraction per plot (rough sheep score)")
    plt.savefig(RUNS_FOLDER / "quick_wool_map.png", dpi=70)
    survey_plots.to_csv(RUNS_FOLDER / "survey_grid.csv", index=False)
