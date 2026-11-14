"""
Laos — "Bad BOMB": find the ~28 defective shells among 1225 unlabeled photos (unsupervised anomaly detection).

USAGE (from this folder)
    python find_defective_shells.py                 # -> test-output.txt (28 names) + runs/ (scores, preview images)
    python find_defective_shells.py --top 40        # write a different number of names

THE IDEA
    Nobody tells us which shells are defective, but defects are RARE (~2%). So: a part of a shell that looks
    unlike the same part on every other shell is suspicious.

        shell #812, patch at row 20 / column 6 (a scratch)   vs the same patch on the 1224 other shells
        -> even its 5 most similar patches are far away      -> high "strangeness"
        shell #44, same patch (normal paint)                 -> hundreds of near-identical patches -> low

PIPELINE
    1. find the shell (green pixels), cut out the same frame around it in every image   -> (1225, 448, 160, 3)
       and normalise the lighting (each crop to the same mean / contrast)
    2. pretrained ResNet18 (ImageNet) describes every 8x8-pixel patch with numbers       -> (1225, 56, 20, 96)
    3. for every patch: distance to its 5 nearest patches at the same (or a neighbouring)
       position in OTHER images; image score = its strangest patch                       -> (1225,)
    4. the 28 highest scores -> test-output.txt; preview images of the top ones are saved to check by eye
"""

import argparse
import glob
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

TASK_FOLDER = Path(__file__).resolve().parent
IMAGES_FOLDER = TASK_FOLDER / "data" / "task_laos" / "images"
RUNS_FOLDER = TASK_FOLDER / "runs"
OUTPUT_FILE = TASK_FOLDER / "test-output.txt"

CROP_HEIGHT, CROP_WIDTH = 448, 160      # frame around the shell, incl. the dark base below the green body
CROP_TOP_MARGIN = 8                     # pixels kept above the shell tip
FEATURE_DIMENSIONS = 96                 # patch descriptions are compressed to 96 numbers (memory: ~0.5 GB total)
NEAREST_NEIGHBOURS = 20                 # "strangeness" = mean distance to the 20 most similar patches elsewhere.
                                        # Must be larger than the number of shells sharing one defect type
                                        # (~9 each for holes / notches / scratches): with k = 5 a notch found
                                        # 5 other notches as "neighbours" and looked normal.
NUMBER_OF_DEFECTS = 28


# ============================================================================ 1. FIND AND CUT OUT THE SHELL

def list_unique_images():
    """All PNGs except the 16 'image_000X(1).png' files, which are byte-identical copies of image_000X.png."""
    return sorted(path for path in glob.glob(str(IMAGES_FOLDER / "*.png")) if "(1)" not in os.path.basename(path))


def find_shell_box(bgr_image):
    """
    The shell body is the largest green region. In OpenCV's HSV, green hue is about 35-90 (of 0-180).
    Returns (left, top, width, height) of that region, e.g. (203, 17, 101, 347).
    """
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_image, (35, 60, 20), (90, 255, 255))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))   # remove green specks
    count, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask)
    if count < 2:
        return 203, 17, 105, 347                                   # fallback: typical position
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return tuple(int(value) for value in stats[largest, :4])


def crop_shell(bgr_image):
    """
    Same frame for every shell: centred on the green body, starting just above its tip.
    The frame is taller than the green part so the dark base below it is included.
    Then normalise lighting: every colour channel gets mean 0 / std 1 inside the frame,
    so a photo taken in brighter light does not look "strange".
    """
    left, top, width, _ = find_shell_box(bgr_image)
    centre_x = left + width // 2
    padded = cv2.copyMakeBorder(bgr_image, CROP_HEIGHT, CROP_HEIGHT, CROP_WIDTH, CROP_WIDTH, cv2.BORDER_REFLECT)
    crop_top = top - CROP_TOP_MARGIN + CROP_HEIGHT
    crop_left = centre_x - CROP_WIDTH // 2 + CROP_WIDTH
    crop = padded[crop_top:crop_top + CROP_HEIGHT, crop_left:crop_left + CROP_WIDTH].astype(np.float32)
    return (crop - crop.mean(axis=(0, 1))) / (crop.std(axis=(0, 1)) + 1e-6)


# ============================================================================ 2. DESCRIBE EVERY PATCH

class PatchDescriber(torch.nn.Module):
    """
    ResNet18 up to layer2 (patch = 8x8 px, 128 numbers) and layer3 (16x16 px, 256 numbers, upsampled to the
    same grid). Each location is averaged with its 3x3 neighbourhood (a bit of context, robust to 1-2 px shifts),
    then compressed to FEATURE_DIMENSIONS numbers with a fixed random projection (keeps distances roughly equal).
    """

    def __init__(self):
        super().__init__()
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        self.stem = torch.nn.Sequential(network.conv1, network.bn1, network.relu, network.maxpool, network.layer1)
        self.layer2, self.layer3 = network.layer2, network.layer3
        generator = torch.Generator().manual_seed(0)
        self.register_buffer("projection", torch.randn(384, FEATURE_DIMENSIONS, generator=generator) / FEATURE_DIMENSIONS ** 0.5)

    @torch.no_grad()
    def forward(self, crops):                                   # crops: (batch, 3, 448, 160)
        layer2_features = self.layer2(self.stem(crops))          # (batch, 128, 56, 20)
        layer3_features = self.layer3(layer2_features)           # (batch, 256, 28, 10)
        layer3_features = torch.nn.functional.interpolate(layer3_features, size=layer2_features.shape[-2:], mode="bilinear")
        features = torch.cat([layer2_features, layer3_features], dim=1)                     # (batch, 384, 56, 20)
        features = torch.nn.functional.avg_pool2d(features, 3, stride=1, padding=1)          # 3x3 context
        features = features.permute(0, 2, 3, 1) @ self.projection                            # (batch, 56, 20, 96)
        return features


def shell_mask_on_grid(image_paths, grid_rows, grid_columns):
    """
    Which grid cells show the shell (not background)? Look at the green body in a sample of crops:
    a cell belongs to the shell if it is green in at least half of them. The dark base under the green body
    is not green, so every column that is green in the lowest green row is extended down to the crop bottom.
    Result: (56, 20) array of True/False.
    """
    green_fraction = np.zeros((grid_rows, grid_columns), np.float32)
    sample_paths = image_paths[::10]
    for path in sample_paths:
        crop = crop_shell(cv2.imread(path))
        crop_bgr = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hue = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
        is_green = ((hue >= 35) & (hue <= 90)).astype(np.float32)
        green_fraction += cv2.resize(is_green, (grid_columns, grid_rows), interpolation=cv2.INTER_AREA)
    shell_cells = green_fraction / len(sample_paths) >= 0.5
    lowest_green_row = np.where(shell_cells.any(axis=1))[0].max()
    shell_cells[lowest_green_row:, shell_cells[lowest_green_row]] = True
    shell_cells = cv2.erode(shell_cells.astype(np.uint8), np.ones((1, 3), np.uint8)).astype(bool)  # 1 cell away from the edge
    return shell_cells


def describe_all_shells(image_paths):
    cache_path = RUNS_FOLDER / "patch_features.npy"     # 0.5 GB; lets us try new scoring ideas in seconds
    if cache_path.exists():
        return np.load(cache_path)
    torch.set_num_threads(os.cpu_count() or 4)
    describer = PatchDescriber()
    all_features = []
    for batch_start in range(0, len(image_paths), 32):
        crops = [crop_shell(cv2.imread(path))[:, :, ::-1].copy() for path in image_paths[batch_start:batch_start + 32]]  # BGR -> RGB
        batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2)
        all_features.append(describer(batch).numpy().astype(np.float32))
        print(f"  described {min(batch_start + 32, len(image_paths))}/{len(image_paths)} shells", end="\r")
    print()
    features = np.concatenate(all_features)                       # (1225, 56, 20, 96)
    np.save(cache_path, features)
    return features


# ============================================================================ 3. HOW STRANGE IS EVERY PATCH?

def patch_strangeness(features, shell_cells):
    """
    For each grid position (row, column): compare every shell's patch with the patches of ALL OTHER shells
    at that position and its 8 neighbouring positions (tolerates small misalignment).
    strangeness = mean distance to the NEAREST_NEIGHBOURS closest ones.

    Distances via matrix product: |a - b|^2 = |a|^2 + |b|^2 - 2 a.b  (fast, no huge intermediate arrays).
    """
    number_of_images, grid_rows, grid_columns, _ = features.shape
    strangeness = np.zeros((number_of_images, grid_rows, grid_columns), np.float32)
    image_ids = np.arange(number_of_images)
    for row in range(grid_rows):
        for column in range(grid_columns):
            if not shell_cells[row, column]:
                continue                                                                # background: not scored (stays 0)
            queries = features[:, row, column]                                          # (1225, 96)
            neighbour_rows = slice(max(row - 1, 0), row + 2)
            neighbour_columns = slice(max(column - 1, 0), column + 2)
            candidates = features[:, neighbour_rows, neighbour_columns].reshape(number_of_images, -1, features.shape[-1])
            positions_per_image = candidates.shape[1]
            candidates = candidates.reshape(-1, features.shape[-1])                      # (1225 * up to 9, 96)
            candidate_owner = np.repeat(image_ids, positions_per_image)
            squared_distances = ((queries ** 2).sum(1)[:, None] + (candidates ** 2).sum(1)[None, :]
                                 - 2 * queries @ candidates.T)
            squared_distances[image_ids[:, None] == candidate_owner[None, :]] = np.inf   # ignore the shell itself
            nearest = np.partition(squared_distances, NEAREST_NEIGHBOURS, axis=1)[:, :NEAREST_NEIGHBOURS]
            strangeness[:, row, column] = np.sqrt(np.maximum(nearest, 0)).mean(axis=1)
        print(f"  scored row {row + 1}/{grid_rows}", end="\r")
    print()
    return strangeness


# ============================================================================ 3b. COLOUR CHECK: ORANGE / RUST ON GREEN PAINT

ORANGE_PIXEL_THRESHOLD = 20   # normal shells: <= 8 orange pixels (tiny paint dots); holes: 25-31; scratches: 100-385


def count_orange_pixels_on_shell(bgr_image):
    """
    Holes have an orange rim and scratches are orange streaks; intact green paint has (almost) no orange.
    1. shell interior = convex hull of the green body, shrunk by 3 px (so the brown background never counts)
    2. orange = hue 5-28 (OpenCV scale 0-180), saturation >= 90, brightness >= 70
    Example counts: normal shell 0-8, hole 25-31, scratch 100-385.
    """
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv_image, (35, 60, 20), (90, 255, 255))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask)
    if count < 2:
        return 0
    shell_body = ((labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(shell_body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shell_interior = np.zeros_like(shell_body)
    cv2.drawContours(shell_interior, [cv2.convexHull(max(contours, key=cv2.contourArea))], -1, 255, thickness=-1)
    shell_interior = cv2.erode(shell_interior, np.ones((7, 7), np.uint8))
    orange_mask = cv2.inRange(hsv_image, (5, 90, 70), (28, 255, 255)) & shell_interior
    return int((orange_mask > 0).sum())


# ============================================================================ 4. RANK, SAVE, PREVIEW

def save_preview(image_paths, ranking, strangeness, count, output_path):
    """Crops of the top-ranked shells with their strangeness map in red, to check the result by eye."""
    tiles = []
    for rank, image_index in enumerate(ranking[:count]):
        crop = cv2.imread(image_paths[image_index])
        left, top, width, _ = find_shell_box(crop)
        x0, y0 = max(left + width // 2 - CROP_WIDTH // 2, 0), max(top - CROP_TOP_MARGIN, 0)
        crop = crop[y0:y0 + CROP_HEIGHT, x0:x0 + CROP_WIDTH]
        heat = cv2.resize(strangeness[image_index], (crop.shape[1], crop.shape[0]))
        heat = (255 * (heat - strangeness.min()) / (strangeness.max() - strangeness.min() + 1e-6)).astype(np.uint8)
        overlay = cv2.addWeighted(crop, 0.65, cv2.applyColorMap(heat, cv2.COLORMAP_JET), 0.35, 0)
        tile = np.hstack([crop, overlay])
        cv2.putText(tile, f"{rank + 1}. {Path(image_paths[image_index]).stem[-4:]}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        tiles.append(tile)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    cv2.imwrite(str(output_path), np.vstack(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=NUMBER_OF_DEFECTS)
    arguments = parser.parse_args()
    RUNS_FOLDER.mkdir(exist_ok=True)

    image_paths = list_unique_images()
    print(f"{len(image_paths)} unique images")
    features = describe_all_shells(image_paths)
    shell_cells = shell_mask_on_grid(image_paths, *features.shape[1:3])
    print(f"shell mask: {shell_cells.sum()} of {shell_cells.size} grid cells")
    cv2.imwrite(str(RUNS_FOLDER / "shell_mask.png"), cv2.resize(shell_cells.astype(np.uint8) * 255, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_NEAREST))
    strangeness_cache = RUNS_FOLDER / "strangeness_maps.npy"   # delete it after changing the scoring settings
    if strangeness_cache.exists():
        strangeness = np.load(strangeness_cache)
    else:
        strangeness = patch_strangeness(features, shell_cells)                    # background cells stay 0
        np.save(strangeness_cache, strangeness)

    image_scores = strangeness.reshape(len(image_paths), -1).max(axis=1)        # an image is as strange as its strangest patch
    orange_counts = np.array([count_orange_pixels_on_shell(cv2.imread(path)) for path in image_paths])

    # Final ranking: shells with clear orange damage first (most orange first), then the rest by patch strangeness.
    # Sorting key example: (has_orange=True, 385, ...) comes before (True, 26, ...) before (False, 0, strangeness 3.1)
    has_orange_damage = orange_counts >= ORANGE_PIXEL_THRESHOLD
    ranking = sorted(range(len(image_paths)),
                     key=lambda index: (has_orange_damage[index], orange_counts[index] if has_orange_damage[index] else 0, image_scores[index]),
                     reverse=True)
    print(f"{has_orange_damage.sum()} shells with clear orange damage; the rest of the {arguments.top} come from patch strangeness")

    with open(RUNS_FOLDER / "all_scores.csv", "w") as scores_file:
        scores_file.write("rank,image,orange_pixels,patch_strangeness\n")
        for rank, image_index in enumerate(ranking):
            scores_file.write(f"{rank + 1},{Path(image_paths[image_index]).name},{orange_counts[image_index]},{image_scores[image_index]:.4f}\n")

    top_names = [Path(image_paths[index]).name for index in ranking[:arguments.top]]
    OUTPUT_FILE.write_text("\n".join(top_names) + "\n")
    print(f"wrote {len(top_names)} names to {OUTPUT_FILE}")
    save_preview(image_paths, ranking, strangeness, 32, RUNS_FOLDER / "top_ranked_preview.jpg")


if __name__ == "__main__":
    main()
