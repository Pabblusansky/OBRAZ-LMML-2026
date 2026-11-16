"""Better counter: a small convolutional 'head' on top of the cached ResNet18 cell features.

Why better than ridge (04): ridge looks at ONE cell at a time. A 3x3 convolution looks at a cell AND its
8 neighbours, so it can tell 'half a sheep at the edge of a crowd' from 'a bright patch of grass'.
  input  (batch, 384, 38, 38)  cell features
  conv 3x3 -> 128 -> conv 3x3 -> 64 -> conv 1x1 -> 1
  output (batch, 1, 38, 38)    predicted sheep per cell; photo count = sum of the map
Loss: per-cell squared error (where are the sheep) + small |sum - true count| term (how many).
Augmentation: random left-right / up-down flips of the feature grid and the target grid together.
Usage:  python 05_conv_head_counts.py [--validate]   (--validate: hold out every 5th photo and report the error)
"""
from pathlib import Path
import sys
from importlib import import_module
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASK_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_FOLDER))
count_and_map = import_module("04_count_and_map")      # reuse sheep_per_cell and the folder paths
survey_grid_positions = import_module("02_quick_map").survey_grid_positions
DATA_FOLDER, RUNS_FOLDER = count_and_map.DATA_FOLDER, count_and_map.RUNS_FOLDER
EPOCHS, BATCH_SIZE = 60, 16
torch.set_num_threads(6)


def make_counting_head():
    return nn.Sequential(nn.Conv2d(384, 128, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(64, 1, 1))


def train_counting_head(cell_features, density_targets, seed=0):
    torch.manual_seed(seed)
    random_generator = np.random.default_rng(seed)
    counting_head = make_counting_head()
    optimizer = torch.optim.Adam(counting_head.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(EPOCHS):
        shuffled = random_generator.permutation(len(cell_features))
        for start in range(0, len(shuffled), BATCH_SIZE):
            batch_indices = shuffled[start:start + BATCH_SIZE]
            feature_batch, target_batch = cell_features[batch_indices].float(), density_targets[batch_indices]
            if random_generator.random() < 0.5:
                feature_batch, target_batch = feature_batch.flip(3), target_batch.flip(3)
            if random_generator.random() < 0.5:
                feature_batch, target_batch = feature_batch.flip(2), target_batch.flip(2)
            predicted_density = counting_head(feature_batch)
            cell_loss = ((predicted_density - target_batch) ** 2).mean() * 100
            count_loss = (predicted_density.sum((1, 2, 3)) - target_batch.sum((1, 2, 3))).abs().mean() * 0.01
            optimizer.zero_grad()
            (cell_loss + count_loss).backward()
            optimizer.step()
    return counting_head.eval()


@torch.no_grad()
def count_sheep(counting_head, cell_features):
    return np.concatenate([counting_head(cell_features[start:start + 32].float()).sum((1, 2, 3)).numpy()
                           for start in range(0, len(cell_features), 32)])


if __name__ == "__main__":
    train_counts = pd.read_csv(DATA_FOLDER / "train_counts.csv")
    train_points = pd.read_csv(DATA_FOLDER / "train_points.csv")
    train_features = torch.from_numpy(np.load(RUNS_FOLDER / "train_cell_features.npy")).permute(0, 3, 1, 2)
    density_targets = torch.from_numpy(count_and_map.sheep_per_cell(train_counts.image, train_points)).unsqueeze(1)
    true_counts = train_counts["count"].to_numpy()

    if "--validate" in sys.argv:
        is_validation = np.arange(len(true_counts)) % 5 == 0
        head = train_counting_head(train_features[~is_validation], density_targets[~is_validation])
        predicted = count_sheep(head, train_features[is_validation])
        print(f"validation MAE {np.abs(predicted - true_counts[is_validation]).mean():.2f} sheep on {is_validation.sum()} photos")
        sys.exit()

    counting_head = train_counting_head(train_features, density_targets)
    torch.save(counting_head.state_dict(), RUNS_FOLDER / "counting_head.pt")
    survey_features = torch.from_numpy(np.load(RUNS_FOLDER / "survey_cell_features.npy")).permute(0, 3, 1, 2)
    survey_plots = pd.read_csv(DATA_FOLDER / "survey_plots.csv")
    survey_plots["raw_count"] = count_sheep(counting_head, survey_features)
    survey_plots["count"] = np.clip(np.round(survey_plots.raw_count), 0, None).astype(int)
    survey_plots[["image", "count"]].to_csv(TASK_FOLDER / "counts.csv", index=False)
    survey_plots["row"], survey_plots["column"] = survey_grid_positions(survey_plots)
    survey_plots.to_csv(RUNS_FOLDER / "survey_counts_on_grid.csv", index=False)

    count_grid = np.zeros((9, 37), int)
    count_grid[survey_plots.row, survey_plots.column] = survey_plots["count"]
    print(np.array2string(count_grid, max_line_width=250))
    plt.figure(figsize=(18, 5))
    plt.imshow(count_grid, cmap="gray")
    plt.colorbar()
    plt.title("sheep per plot, conv-head counter (north up)")
    plt.savefig(RUNS_FOLDER / "census_map_conv.png", dpi=70)
