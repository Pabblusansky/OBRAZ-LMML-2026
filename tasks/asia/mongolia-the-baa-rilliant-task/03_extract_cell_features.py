"""Turn every photo into a grid of 'cell descriptions' with a pretrained ResNet18 (no training here).

600x600 photo -> ResNet18 layer2 (stride 8, 128 numbers) + layer3 (stride 16, 256 numbers)
-> both brought to a 38x38 grid -> shape (38, 38, 384): one 384-number description per 16x16-pixel cell.
Saved as float16 in runs/ so the regression step can be re-run quickly.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

TASK_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = TASK_FOLDER / "data" / "task_mongolia"
RUNS_FOLDER = TASK_FOLDER / "runs"
GRID_SIZE = 38          # 600 px / 16 px per cell = 37.5 -> 38 cells
torch.set_num_threads(4)  # share the CPU with the other session's training

resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def cell_features(image_paths):
    """Returns shape (num_images, 38, 38, 384) float16."""
    pixel_batch = torch.stack([torch.from_numpy(np.asarray(Image.open(path).convert("RGB"))).permute(2, 0, 1)
                               for path in image_paths]).float() / 255.0
    pixel_batch = (pixel_batch - IMAGENET_MEAN) / IMAGENET_STD
    stem_output = resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(pixel_batch))))
    layer2_output = resnet.layer2(resnet.layer1(stem_output))        # (B, 128, 75, 75)
    layer3_output = resnet.layer3(layer2_output)                     # (B, 256, 38, 38)
    layer2_on_grid = F.adaptive_avg_pool2d(layer2_output, GRID_SIZE)  # average 2x2 cells -> (B, 128, 38, 38)
    combined = torch.cat([layer2_on_grid, layer3_output], dim=1)     # (B, 384, 38, 38)
    return combined.permute(0, 2, 3, 1).numpy().astype(np.float16)


for split_name, csv_name in [("train", "train_counts.csv"), ("survey", "survey_plots.csv")]:
    image_names = pd.read_csv(DATA_FOLDER / csv_name).image.tolist()
    batches = [cell_features([DATA_FOLDER / split_name / name for name in image_names[start:start + 8]])
               for start in range(0, len(image_names), 8)]
    all_features = np.concatenate(batches)
    np.save(RUNS_FOLDER / f"{split_name}_cell_features.npy", all_features)
    print(split_name, all_features.shape, f"{all_features.nbytes / 1e6:.0f} MB")
