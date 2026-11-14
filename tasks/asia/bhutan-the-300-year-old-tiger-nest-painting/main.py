"""
Bhutan — "The 300-Year Old Tiger Nest Painting": put shuffled vertical image strips back in order.

USAGE
    python main.py              # reads strips from the folder "input" (next to main.py, else in the working folder)
    python main.py some_folder  # explicit input folder (for local testing)
    -> writes reconstructed_image.png and reconstructed_order.json next to main.py

THE IDEA IN ONE PICTURE
    A cut separates two pixel columns that were neighbours in the original, so they are almost identical:

        strip A, last column  :  120 121 119 200 ...      (top to bottom)
        strip B, first column :  122 120 118 201 ...      -> tiny difference -> B goes right after A
        strip C, first column :   40 210  90  15 ...      -> big difference  -> C does not

PIPELINE (each step is a section below)
    1. load all strips                                        192 strips of 1024 x 8 pixels
    2. cost[i, j] = how badly strip j fits right after strip i (192 x 192 numbers),
       then normalised: "how much better is this join than the alternatives?"
    3. chain the strips: take the most convincing joins first, never giving a strip two right
       neighbours, two left neighbours, or closing a loop -> one chain; then make sure the chain
       starts at the true left edge (close it into a ring, cut at the least convincing join)
    4. glue the strips in that order, save the image and the order

Runs in ~1 s for 192 strips (time limit: 20 s). Needs numpy and opencv-python.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ============================================================================ 0. SETTINGS

INPUT_FOLDER_NAME = "input"
OUTPUT_IMAGE_NAME = "reconstructed_image.png"
OUTPUT_ORDER_NAME = "reconstructed_order.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SCRIPT_FOLDER = Path(__file__).resolve().parent


# ============================================================================ 1. LOAD THE STRIPS

def find_input_folder():
    """Command-line folder if given, else 'input' next to main.py, else 'input' in the working folder."""
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for candidate in (SCRIPT_FOLDER / INPUT_FOLDER_NAME, Path.cwd() / INPUT_FOLDER_NAME):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no '{INPUT_FOLDER_NAME}' folder next to main.py or in {Path.cwd()}")


def load_strips(input_folder):
    """
    Returns (strip_names, original_strips, colour_strips):
      original_strips: exactly as stored (kept untouched so the output image is pixel-exact)
      colour_strips  : float32 copies with 3 channels, shape (number_of_strips, height, width, 3), for computing
    Names are sorted only so that runs are repeatable; the names carry no position information.
    """
    strip_paths = sorted(path for path in Path(input_folder).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    strip_names = [path.name for path in strip_paths]
    original_strips = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in strip_paths]

    colour_strips = []
    for strip in original_strips:
        if strip.ndim == 2:                       # grayscale -> 3 identical channels
            strip = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)
        colour_strips.append(strip[:, :, :3].astype(np.float32))  # drop alpha if present
    return strip_names, original_strips, np.stack(colour_strips)


# ============================================================================ 2. HOW WELL DOES STRIP j FIT RIGHT AFTER STRIP i?

def compute_join_costs(colour_strips):
    """
    cost[i, j] = mismatch when strip j is placed directly to the right of strip i (lower = better fit).

    Instead of only comparing A's last column with B's first column, we PREDICT across the cut:
    images change smoothly, so if brightness goes 100 -> 110 over A's last two columns,
    the next column (B's first) should be about 120.

        predicted B-first  = 2 * A-last  - A-second-to-last        (continue A's trend to the right)
        predicted A-last   = 2 * B-first - B-second               (continue B's trend to the left)
        cost = sum of |prediction - actual| over all pixels and colour channels, in both directions

    Example, one pixel: A columns ... 100, 110 | B columns 121, 131 ...
        prediction from A: 2*110 - 100 = 120  vs actual 121  -> error 1
        prediction from B: 2*121 - 131 = 111  vs actual 110  -> error 1   -> cost 2 (excellent fit)
    """
    last_columns = colour_strips[:, :, -1, :]            # (N, height, 3): right edge of every strip
    second_last_columns = colour_strips[:, :, -2, :]
    first_columns = colour_strips[:, :, 0, :]             # (N, height, 3): left edge of every strip
    second_columns = colour_strips[:, :, 1, :]

    predicted_next_columns = 2 * last_columns - second_last_columns      # what should come right of each strip
    predicted_previous_columns = 2 * first_columns - second_columns      # what should come left of each strip

    number_of_strips = len(colour_strips)
    join_costs = np.empty((number_of_strips, number_of_strips), dtype=np.float64)
    for left_index in range(number_of_strips):           # one row at a time keeps memory small
        error_predicting_right = np.abs(predicted_next_columns[left_index][None] - first_columns).sum(axis=(1, 2))
        error_predicting_left = np.abs(predicted_previous_columns - last_columns[left_index][None]).sum(axis=(1, 2))
        join_costs[left_index] = error_predicting_right + error_predicting_left
    np.fill_diagonal(join_costs, np.inf)                  # a strip cannot follow itself
    return join_costs


def normalise_join_costs(join_costs):
    """
    Turn raw costs into "how much better is this join than the alternatives?"

        normalised[i, j] = cost[i, j] / (best cost of any OTHER join involving i on its right or j on its left)

    Why: raw costs are big in busy areas and small in flat areas.
        real join in a busy area : cost 900, best alternative 4000 -> 0.22  (clearly right)
        join between flat borders: cost 300, best alternative  320 -> 0.94  (could be anything)
    With raw costs the flat join looks better (300 < 900); normalised, the busy one correctly wins.
    """
    def best_alternative_per_row(costs):
        two_smallest = np.partition(costs, 1, axis=1)[:, :2]            # (N, 2): best and second best per row
        is_the_best = costs == two_smallest[:, :1]
        return np.where(is_the_best, two_smallest[:, 1:2], two_smallest[:, :1])

    alternative_for_left_strip = best_alternative_per_row(join_costs)          # other right neighbours of i
    alternative_for_right_strip = best_alternative_per_row(join_costs.T).T     # other left neighbours of j
    return join_costs / (np.minimum(alternative_for_left_strip, alternative_for_right_strip) + 1e-9)


# ============================================================================ 3. CHAIN THE STRIPS INTO ONE ORDER

def order_strips(join_costs):
    """
    Greedy chaining, best joins first (like assembling a puzzle by doing the obvious pieces first):

      all 192*191 possible joins, sorted from best (lowest cost) to worst
      for each join "i then j":
          accept it if  - i has no right neighbour yet
                        - j has no left neighbour yet
                        - i and j are not already in the same chain (that would close a loop)
      stop when 191 joins are accepted -> all strips form one chain

    Example with 4 strips, joins sorted best first:
        2->0  accept                      chains: [2,0]  [1]  [3]
        0->3  accept                      chains: [2,0,3]  [1]
        3->2  reject (would close the loop 2->0->3->2)
        0->1  reject (0 already has a right neighbour)
        1->2  accept                      chains: [1,2,0,3]   -> 3 joins = done, order 1, 2, 0, 3
    The leftmost strip is the one that never received a left neighbour.
    """
    number_of_strips = len(join_costs)
    right_neighbour = [-1] * number_of_strips
    has_left_neighbour = [False] * number_of_strips
    chain_id = list(range(number_of_strips))   # union-find: which chain each strip belongs to

    def find_chain(strip_index):
        while chain_id[strip_index] != strip_index:
            chain_id[strip_index] = chain_id[chain_id[strip_index]]
            strip_index = chain_id[strip_index]
        return strip_index

    joins_from_best_to_worst = np.argsort(join_costs, axis=None)
    accepted_joins = 0
    for flat_index in joins_from_best_to_worst:
        left_index, right_index = divmod(int(flat_index), number_of_strips)
        if left_index == right_index or right_neighbour[left_index] != -1 or has_left_neighbour[right_index]:
            continue
        if find_chain(left_index) == find_chain(right_index):
            continue
        right_neighbour[left_index] = right_index
        has_left_neighbour[right_index] = True
        chain_id[find_chain(left_index)] = find_chain(right_index)
        accepted_joins += 1
        if accepted_joins == number_of_strips - 1:
            break

    leftmost_strip = has_left_neighbour.index(False)
    order = [leftmost_strip]
    while right_neighbour[order[-1]] != -1:
        order.append(right_neighbour[order[-1]])
    return cut_ring_at_weakest_join(order, join_costs)


def cut_ring_at_weakest_join(order, join_costs):
    """
    The chain may start in the wrong place: if the picture's right edge happens to fit its left edge
    well, the chain "wraps around" and one real join in the middle is left out. Every strip is then
    shifted (e.g. true positions 35, 36, ..., 191, 0, 1, ..., 34), and almost none is in its exact spot.

    Fix: close the chain into a ring (last strip -> first strip), then cut the ring at the join
    that is least convincing (highest normalised cost). The two outer edges of a picture never
    really continue each other, so that is where the cut belongs.
    """
    ring_join_costs = [join_costs[order[k], order[(k + 1) % len(order)]] for k in range(len(order))]
    cut_after_position = int(np.argmax(ring_join_costs))
    return order[cut_after_position + 1:] + order[:cut_after_position + 1]


# ============================================================================ 4. GLUE AND SAVE

def save_results(strip_names, original_strips, order, output_folder=SCRIPT_FOLDER):
    reconstructed_image = np.hstack([original_strips[index] for index in order])  # side by side, left to right
    cv2.imwrite(str(Path(output_folder) / OUTPUT_IMAGE_NAME), reconstructed_image)
    with open(Path(output_folder) / OUTPUT_ORDER_NAME, "w") as order_file:
        json.dump([strip_names[index] for index in order], order_file, indent=4)
    return reconstructed_image


def reconstruct(input_folder):
    """Steps 1-3: returns strip names, original strips and the order (list of indices, left to right)."""
    strip_names, original_strips, colour_strips = load_strips(input_folder)
    if len(strip_names) < 2:
        return strip_names, original_strips, list(range(len(strip_names)))
    return strip_names, original_strips, order_strips(normalise_join_costs(compute_join_costs(colour_strips)))


def main():
    input_folder = find_input_folder()
    strip_names, original_strips, order = reconstruct(input_folder)
    reconstructed_image = save_results(strip_names, original_strips, order)
    print(f"{len(strip_names)} strips from {input_folder} -> image {reconstructed_image.shape[1]}x{reconstructed_image.shape[0]}")
    print(f"saved {OUTPUT_IMAGE_NAME} and {OUTPUT_ORDER_NAME} in {SCRIPT_FOLDER}")


if __name__ == "__main__":
    main()
