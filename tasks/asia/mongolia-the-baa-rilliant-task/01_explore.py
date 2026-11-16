"""Look at the data: a few training photos with their sheep dots, and the survey plot positions on a map."""
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TASK_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = TASK_FOLDER / "data" / "task_mongolia"
RUNS_FOLDER = TASK_FOLDER / "runs"
RUNS_FOLDER.mkdir(exist_ok=True)

train_counts = pd.read_csv(DATA_FOLDER / "train_counts.csv")
train_points = pd.read_csv(DATA_FOLDER / "train_points.csv")
survey_plots = pd.read_csv(DATA_FOLDER / "survey_plots.csv")

# 1) 8 training photos (sorted by count, spread from few to many) with a red dot on every labelled sheep
example_names = train_counts.sort_values("count").image.iloc[[0, 40, 80, 120, 160, 200, 240, 279]]
figure, axes = plt.subplots(2, 4, figsize=(20, 10))
for axis, image_name in zip(axes.ravel(), example_names):
    rgb_image = cv2.cvtColor(cv2.imread(str(DATA_FOLDER / "train" / image_name)), cv2.COLOR_BGR2RGB)
    sheep_dots = train_points[train_points.image == image_name]
    axis.imshow(rgb_image)
    axis.scatter(sheep_dots.x, sheep_dots.y, s=8, c="red")
    axis.set_title(f"{image_name}: {len(sheep_dots)} sheep")
    axis.axis("off")
plt.tight_layout()
plt.savefig(RUNS_FOLDER / "train_examples.png", dpi=60)

# 2) where the 333 survey plots are (metres); the colour is just the file number for now
figure, axis = plt.subplots(figsize=(16, 6))
axis.scatter(survey_plots.x_m, survey_plots.y_m, s=30, c=range(len(survey_plots)), cmap="viridis")
axis.set_aspect("equal")
axis.set_title("survey plot positions (colour = file order)")
plt.savefig(RUNS_FOLDER / "survey_positions.png", dpi=70)
print("saved", RUNS_FOLDER / "train_examples.png", RUNS_FOLDER / "survey_positions.png")
