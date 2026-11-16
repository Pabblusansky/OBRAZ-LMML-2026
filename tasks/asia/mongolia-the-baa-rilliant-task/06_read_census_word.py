"""Read the word hidden in the census: the 9x37 plot grid is a dot-matrix display.

Plots inside a letter stroke ('ink') hold more sheep (~50) than background plots (~32).
Layout found from the map: 6 letters, each 5 columns x 7 rows, 1 empty column between letters,
1 empty border row/column around everything:  1 + 6*(5+1) = 37 columns, 1 + 7 + 1 = 9 rows.
Example (letter 6): left column 53, 68, 56, 49, 56, 24(miss), 63 and bottom row 63 54 53 56 52 -> 'L'.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_FOLDER = Path(__file__).resolve().parent
RUNS_FOLDER = TASK_FOLDER / "runs"
INK_THRESHOLD = 44          # between the background level (~32) and the ink level (~50)

survey_counts = pd.read_csv(RUNS_FOLDER / "survey_counts_on_grid.csv")   # from 05_conv_head_counts.py
count_grid = np.zeros((9, 37))
count_grid[survey_counts.row, survey_counts.column] = survey_counts.raw_count

print("ink map (# = more than", INK_THRESHOLD, "sheep):")
for grid_row in count_grid:
    print("  " + "".join("#" if count > INK_THRESHOLD else "." for count in grid_row))

figure, axes = plt.subplots(2, 1, figsize=(16, 7))
axes[0].imshow(count_grid, cmap="gray")
axes[0].set_title("predicted sheep per plot (grey level = count)")
axes[1].imshow(count_grid > INK_THRESHOLD, cmap="gray")
axes[1].set_title(f"plots with more than {INK_THRESHOLD} sheep")
for axis in axes:
    for letter_start in range(1, 37, 6):    # outline each 5x7 letter cell
        axis.add_patch(plt.Rectangle((letter_start - 0.5, 0.5), 5, 7, fill=False, edgecolor="red", linewidth=1.5))
    axis.set_xticks([])
    axis.set_yticks([])
plt.tight_layout()
plt.savefig(RUNS_FOLDER / "census_word.png", dpi=80)
print("saved", RUNS_FOLDER / "census_word.png")
