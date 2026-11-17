"""
India — "Twins Separated at Birth": find all pairs of images that come from the SAME source photograph.

USAGE
    python main.py                  # looks for images (or .zip archives) inside the current folder
    python main.py path/to/input    # a folder or a .zip with the images
    -> writes output.csv (columns: image_a,image_b) into the current working directory

THE PROBLEM IN ONE PICTURE
    img_A.jpg  (612x408)  ──resize, crop, rotate 3°, re-save, brighten──▶  img_B.jpg (544x363)   = DUPLICATE
    img_C.jpg  same monument, same angle, photographed 2 seconds later      = NOT a duplicate (the trap)

PIPELINE (each step is a section below)
    1. find the input images
    2. prepare every image once: tiny fingerprint + keypoints
    3. CANDIDATES: for each image, its 10 most similar images by fingerprint      (cheap, ~8 s for 4000 images)
       -> 4057 images give ~33 000 candidate pairs instead of 8.2 million possible pairs
    4. VERIFY each candidate: match keypoints, fit one geometric transform, align the images,
       compare fine details                                                         (exact, ~50 s)
    5. complete groups: if A~B and B~C, then A~C as well
    6. write output.csv

RESULT ON THE TRAINING SET (4057 images, 1399 true pairs): see check_f1.py.
"""

import csv
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ============================================================================ 0. SETTINGS
# Thresholds were chosen by measuring F1 on the training set (check_f1.py --sweep).
# Results were stable over a range of values, so these sit in the middle of that range.

NEIGHBOURS_PER_IMAGE = 10          # candidates per image; the true twin is in the top 10 for 99.9% of images
FINGERPRINT_IMAGE_SIZE = 128       # longest side (px) of the copy used for fingerprints
MATCHING_IMAGE_SIZE = 512          # longest side (px) of the copy used for keypoint matching
KEYPOINTS_PER_IMAGE = 1000         # ORB keypoints detected per image
MATCH_RATIO_TEST = 0.8             # a keypoint match is kept only if clearly better than the 2nd best (Lowe's ratio test)
RANSAC_MAX_ERROR_PX = 4.0          # a match "agrees" with the transform if it lands within 4 px

MINIMUM_INLIERS = 40               # at least 40 keypoint matches must agree on one transform
MAXIMUM_ALIGNMENT_ERROR_PX = 1.3   # same photo -> points land within ~0.01-0.8 px; a moved camera -> more
MINIMUM_DETAIL_CORRELATION = 0.6   # aligned fine details (clouds, leaves, people) must agree

USE_GROUP_COMPLETION = False       # add A~C when A~B and B~C were found. Measured on train: it HURTS
                                   # (F1 0.957 -> 0.949): one false match merges two groups into many false pairs.

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OUTPUT_FILE_NAME = "output.csv"


# ============================================================================ 1. FIND THE INPUT IMAGES

ARCHIVE_NAME_ENDINGS = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".gz", ".bz2", ".xz")


def find_input_images(input_location=None):
    """
    Return a list of image paths.
    - input_location given: a folder or an archive file.
    - nothing given: the current working folder.
    Every archive found (also archives inside archives) is unpacked into a temporary folder and searched too.

    Example: working folder = [main.py, test-input.zip]
             -> unpack test-input.zip to /tmp/twins_input_x/ -> find /tmp/twins_input_x/test/img_000001.jpg, ...
    """
    start_location = Path(input_location) if input_location is not None else Path.cwd()
    folders_to_search = [start_location] if start_location.is_dir() else []
    archives_to_unpack = [start_location] if start_location.is_file() else []
    image_paths = []

    while folders_to_search or archives_to_unpack:
        if archives_to_unpack:
            extraction_folder = extract_archive(archives_to_unpack.pop())
            if extraction_folder is not None:
                folders_to_search.append(extraction_folder)
            continue
        # os.walk with followlinks=True also enters linked folders (Path.rglob silently skips them)
        for current_folder, _, file_names in os.walk(folders_to_search.pop(), followlinks=True):
            if "__MACOSX" in Path(current_folder).parts:  # macOS zip junk: "._img.jpg" files that are not images
                continue
            for file_name in file_names:
                file_path = Path(current_folder) / file_name
                if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(file_path)
                elif file_name.lower().endswith(ARCHIVE_NAME_ENDINGS):
                    archives_to_unpack.append(file_path)

    if not image_paths:
        raise RuntimeError("no images found.\n" + describe_surroundings(start_location))
    return sorted(set(image_paths))


def extract_archive(archive_path):
    """
    Unpack an archive into a new temporary folder; return that folder, or None if it cannot be unpacked.
    The type is detected from the file's CONTENT, not its name: a file called .zip may really be a .tar.gz.
    Failure prints a warning instead of crashing, so other inputs can still be used.
    """
    extraction_folder = Path(tempfile.mkdtemp(prefix="twins_input_"))
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(extraction_folder)
        elif tarfile.is_tarfile(archive_path):          # .tar, .tar.gz, .tar.bz2, .tar.xz (compression auto-detected)
            with tarfile.open(archive_path) as archive:
                archive.extractall(extraction_folder)
        else:
            shutil.unpack_archive(str(archive_path), str(extraction_folder))
        print(f"unpacked {archive_path}")
        return extraction_folder
    except Exception as error:
        print(f"WARNING: could not unpack {archive_path}: {error!r}\n{describe_file(archive_path)}", file=sys.stderr)
        return None


def describe_file(file_path):
    """Size and first bytes of a file. First bytes reveal the real type: b'PK' = zip, b'\\x1f\\x8b' = gzip, b'7z' = 7-Zip."""
    with open(file_path, "rb") as opened_file:
        first_bytes = opened_file.read(16)
    description = f"  {file_path}: {os.path.getsize(file_path)} bytes, starts with {first_bytes!r}"
    if first_bytes.startswith(b"version https://"):
        # A tiny text file like "version https://git-lfs.github.com/spec/v1 / oid sha256:... / size ..." is a
        # Git LFS POINTER: the real file was never downloaded (it needs `git lfs pull` on the server).
        description += ("\n    -> this is a Git LFS pointer, NOT the real archive (needs `git lfs pull`). Full content:\n"
                        + Path(file_path).read_text(errors="replace"))
    return description


def describe_surroundings(folder):
    """List what is in the input location (2 levels), to understand an unexpected input from the error message."""
    folder = folder if folder.is_dir() else folder.parent
    lines = [f"contents of {folder}:"]
    for entry in sorted(folder.iterdir())[:50]:
        lines.append(describe_file(entry) if entry.is_file() else f"  {entry}/ ({len(list(entry.iterdir()))} entries)")
    return "\n".join(lines)


# ============================================================================ 2. PREPARE EVERY IMAGE ONCE

@dataclass
class PreparedImage:
    name: str                         # file name written to output.csv, e.g. "img_000004.jpg"
    thumbnail_vector: np.ndarray      # 256 numbers: 16x16 grayscale thumbnail of the image centre, normalised
    colour_histogram: np.ndarray      # 128 numbers: how much of each hue/saturation the image contains
    matching_grayscale: np.ndarray    # grayscale copy, longest side 512 px, used for alignment checks
    keypoint_positions: np.ndarray    # shape (number_of_keypoints, 2): x, y of each keypoint
    keypoint_descriptors: np.ndarray  # shape (number_of_keypoints, 32): 256-bit ORB description of each keypoint


def resize_to_longest_side(image, longest_side):
    """Shrink (never enlarge) so that max(height, width) == longest_side. 612x408 -> 128x85 for 128."""
    height, width = image.shape[:2]
    scale = longest_side / max(height, width)
    if scale >= 1:
        return image
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)  # INTER_AREA = average pixels -> no aliasing


def central_part(image, kept_fraction=0.8):
    """Keep the middle 80%: duplicates are often slightly cropped or rotated (black corners), the centre survives."""
    height, width = image.shape[:2]
    trim_y, trim_x = int(height * (1 - kept_fraction) / 2), int(width * (1 - kept_fraction) / 2)
    return image[trim_y:height - trim_y, trim_x:width - trim_x]


def make_thumbnail_vector(small_grayscale):
    """
    16x16 thumbnail -> 256 numbers -> subtract the mean, divide by the length.
    After that, the dot product of two vectors is their correlation (1 = same pattern),
    and a brightness change (all pixels +20) does not change the vector at all.
    """
    thumbnail = cv2.resize(central_part(small_grayscale), (16, 16), interpolation=cv2.INTER_AREA)
    vector = thumbnail.astype(np.float32).flatten()
    vector -= vector.mean()
    return vector / (np.linalg.norm(vector) + 1e-6)


def make_colour_histogram(small_colour_image):
    """
    Count pixels per (hue, saturation) bin: 16 hue bins x 8 saturation bins = 128 numbers.
    Brightness (V in HSV) is ignored on purpose, so a brightened duplicate keeps the same histogram.
    The square root + normalisation makes the dot product of two histograms a similarity in 0..1.
    """
    hsv_image = cv2.cvtColor(small_colour_image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv_image], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    return np.sqrt(histogram / histogram.sum())


def prepare_image(image_path):
    colour_image = cv2.imread(str(image_path))  # NOTE: OpenCV gives B, G, R channel order
    if colour_image is None:
        return None  # unreadable file -> skipped

    small_colour_image = resize_to_longest_side(colour_image, FINGERPRINT_IMAGE_SIZE)
    small_grayscale = cv2.cvtColor(small_colour_image, cv2.COLOR_BGR2GRAY)
    matching_grayscale = cv2.cvtColor(resize_to_longest_side(colour_image, MATCHING_IMAGE_SIZE), cv2.COLOR_BGR2GRAY)

    # ORB finds corner-like spots (window corners, carvings) and describes each one with 256 bits.
    # The same physical spot gets almost the same 256 bits even after resizing / rotating / brightening.
    keypoints, descriptors = cv2.ORB_create(nfeatures=KEYPOINTS_PER_IMAGE).detectAndCompute(matching_grayscale, None)
    keypoint_positions = np.float32([keypoint.pt for keypoint in keypoints]) if keypoints else np.zeros((0, 2), np.float32)

    return PreparedImage(
        name=Path(image_path).name,
        thumbnail_vector=make_thumbnail_vector(small_grayscale),
        colour_histogram=make_colour_histogram(small_colour_image),
        matching_grayscale=matching_grayscale,
        keypoint_positions=keypoint_positions,
        keypoint_descriptors=descriptors,
    )


# ============================================================================ 3. CANDIDATE PAIRS

def find_candidate_pairs(prepared_images, neighbours_per_image=NEIGHBOURS_PER_IMAGE):
    """
    similarity[i, j] = thumbnail correlation + colour-histogram similarity   (max 2.0)
    For every image keep its `neighbours_per_image` most similar images.
    Pairs are stored as (smaller index, larger index), so (3, 7) and (7, 3) count once.

    Example: image 3's top neighbours are [7, 12, 5, ...] -> candidates (3, 7), (3, 12), (3, 5), ...
    """
    thumbnail_matrix = np.array([image.thumbnail_vector for image in prepared_images])  # (N, 256)
    histogram_matrix = np.array([image.colour_histogram for image in prepared_images])  # (N, 128)
    similarity = thumbnail_matrix @ thumbnail_matrix.T + histogram_matrix @ histogram_matrix.T  # (N, N)
    np.fill_diagonal(similarity, -np.inf)  # an image is not its own candidate

    neighbours_per_image = min(neighbours_per_image, len(prepared_images) - 1)
    if neighbours_per_image < 1:
        return []
    nearest_neighbours = np.argsort(-similarity, axis=1)[:, :neighbours_per_image]  # (N, k) indices, most similar first

    candidate_pairs = {tuple(sorted((image_index, int(neighbour_index))))
                       for image_index, neighbours in enumerate(nearest_neighbours)
                       for neighbour_index in neighbours}
    return sorted(candidate_pairs)


# ============================================================================ 4. VERIFY A CANDIDATE PAIR

@dataclass
class PairMeasurement:
    inliers: int                     # keypoint matches that agree on one transform (same photo: typically 100-600)
    alignment_error_px: float        # median distance between where agreeing points land and where they should be
    detail_correlation: float        # correlation of fine details after alignment (same photo: typically 0.7-0.95)


NO_MATCH = PairMeasurement(inliers=0, alignment_error_px=float("inf"), detail_correlation=0.0)
keypoint_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)  # compares 256-bit descriptors by number of differing bits


def fine_details(grayscale_image):
    """Image minus its blurred self = only small details (edges, texture, clouds), no large brightness areas."""
    as_float = grayscale_image.astype(np.float32)
    return as_float - cv2.GaussianBlur(as_float, (0, 0), 3)


def correlation_inside_mask(first_values, second_values, mask):
    """Pearson correlation of pixel values inside `mask`: 1 = identical pattern, 0 = unrelated."""
    first = first_values[mask].astype(np.float32)
    second = second_values[mask].astype(np.float32)
    first -= first.mean()
    second -= second.mean()
    return float((first * second).sum() / (np.sqrt((first * first).sum() * (second * second).sum()) + 1e-6))


def measure_pair(first_image, second_image):
    """
    Step by step, with typical numbers for a true duplicate:
      a) match descriptors: 1000 vs 1000 keypoints -> 420 confident matches
      b) RANSAC fits ONE transform (scale + rotation + shift) -> 395 of 420 matches agree ("inliers")
      c) alignment error: agreeing points land on average 0.02 px from their partner
      d) warp the first image onto the second, compare fine details -> correlation 0.91
    A different photo of the same monument typically gets: 60 inliers, 2.5 px error, 0.25 detail correlation.
    """
    first_descriptors, second_descriptors = first_image.keypoint_descriptors, second_image.keypoint_descriptors
    if first_descriptors is None or second_descriptors is None or len(first_descriptors) < 8 or len(second_descriptors) < 8:
        return NO_MATCH

    # a) for each keypoint of image 1: best and second-best partner in image 2.
    #    Keep it only if the best is clearly better (ratio test) -> removes ambiguous matches on repeated patterns.
    two_best_matches = keypoint_matcher.knnMatch(first_descriptors, second_descriptors, k=2)
    confident_matches = [best for best, *others in two_best_matches
                         if others and best.distance < MATCH_RATIO_TEST * others[0].distance]
    if len(confident_matches) < 6:
        return NO_MATCH

    points_in_first = first_image.keypoint_positions[[match.queryIdx for match in confident_matches]]
    points_in_second = second_image.keypoint_positions[[match.trainIdx for match in confident_matches]]

    # b) RANSAC: repeatedly pick 2 random matches, compute the transform they imply, count how many other
    #    matches agree within RANSAC_MAX_ERROR_PX; keep the transform with the most agreement.
    #    "Partial affine" = rotation + uniform scale + shift: exactly the edits a duplicate went through.
    transform, inlier_flags = cv2.estimateAffinePartial2D(points_in_first, points_in_second,
                                                          method=cv2.RANSAC, ransacReprojThreshold=RANSAC_MAX_ERROR_PX)
    if transform is None:
        return NO_MATCH
    inlier_flags = inlier_flags.ravel().astype(bool)
    inlier_count = int(inlier_flags.sum())

    # c) where do the agreeing points land? transform = [[a, b, shift_x], [c, d, shift_y]]
    landed_positions = points_in_first[inlier_flags] @ transform[:, :2].T + transform[:, 2]
    alignment_error_px = float(np.median(np.linalg.norm(landed_positions - points_in_second[inlier_flags], axis=1)))

    # d) warp image 1 into image 2's frame; compare only where both images have pixels (minus a 2 px border)
    second_height, second_width = second_image.matching_grayscale.shape
    warped_first = cv2.warpAffine(first_image.matching_grayscale, transform, (second_width, second_height))
    covered_area = cv2.warpAffine(np.ones_like(first_image.matching_grayscale), transform, (second_width, second_height))
    covered_area = cv2.erode(covered_area, np.ones((5, 5), np.uint8)) > 0
    if covered_area.sum() < 500:
        return PairMeasurement(inlier_count, alignment_error_px, 0.0)
    detail_correlation = correlation_inside_mask(fine_details(warped_first), fine_details(second_image.matching_grayscale), covered_area)

    return PairMeasurement(inlier_count, alignment_error_px, detail_correlation)


def is_duplicate(measurement):
    return (measurement.inliers >= MINIMUM_INLIERS
            and measurement.alignment_error_px <= MAXIMUM_ALIGNMENT_ERROR_PX
            and measurement.detail_correlation >= MINIMUM_DETAIL_CORRELATION)


# ============================================================================ 5. COMPLETE GROUPS

def complete_groups(duplicate_pairs, number_of_images):
    """
    Union-find: every image starts in its own group; each duplicate pair merges two groups.
    Then output every pair inside each group.
    Example: found (1, 2) and (2, 3) -> group {1, 2, 3} -> pairs (1, 2), (1, 3), (2, 3).
    """
    group_parent = list(range(number_of_images))

    def find_group_root(image_index):
        while group_parent[image_index] != image_index:
            group_parent[image_index] = group_parent[group_parent[image_index]]  # shortcut for faster later lookups
            image_index = group_parent[image_index]
        return image_index

    for first_index, second_index in duplicate_pairs:
        group_parent[find_group_root(first_index)] = find_group_root(second_index)

    members_by_group = {}
    for image_index in {index for pair in duplicate_pairs for index in pair}:
        members_by_group.setdefault(find_group_root(image_index), []).append(image_index)

    completed_pairs = set()
    for members in members_by_group.values():
        members.sort()
        completed_pairs.update((members[a], members[b]) for a in range(len(members)) for b in range(a + 1, len(members)))
    return sorted(completed_pairs)


# ============================================================================ 6. WHOLE PIPELINE + OUTPUT

def run_parallel(function, items):
    cv2.setNumThreads(1)  # we parallelise ourselves; stop OpenCV from also spawning threads per call
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        return list(pool.map(function, items))


def prepare_and_measure(image_paths):
    """Steps 2-4. Returned separately so check_f1.py can try other thresholds without recomputing."""
    prepared_images = [image for image in run_parallel(prepare_image, image_paths) if image is not None]
    print(f"prepared {len(prepared_images)} images")
    candidate_pairs = find_candidate_pairs(prepared_images)
    print(f"{len(candidate_pairs)} candidate pairs")
    measurements = run_parallel(lambda pair: measure_pair(prepared_images[pair[0]], prepared_images[pair[1]]), candidate_pairs)
    return prepared_images, candidate_pairs, measurements


def select_duplicate_pairs(prepared_images, candidate_pairs, measurements, use_group_completion=USE_GROUP_COMPLETION):
    """Steps 4 (decision) and 5. Returns pairs of file names."""
    duplicate_pairs = [pair for pair, measurement in zip(candidate_pairs, measurements) if is_duplicate(measurement)]
    if use_group_completion:
        duplicate_pairs = complete_groups(duplicate_pairs, len(prepared_images))
    name_pairs = {tuple(sorted((prepared_images[first].name, prepared_images[second].name))) for first, second in duplicate_pairs}
    return sorted(pair for pair in name_pairs if pair[0] != pair[1])  # no self-pairs (also guards duplicate file names)


def write_output_csv(name_pairs, output_path=OUTPUT_FILE_NAME):
    """Header is always written, even with zero pairs (required by the task)."""
    with open(output_path, "w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["image_a", "image_b"])
        writer.writerows(name_pairs)


def main():
    input_location = sys.argv[1] if len(sys.argv) > 1 else None
    image_paths = find_input_images(input_location)
    print(f"found {len(image_paths)} images")

    name_pairs = []
    if len(image_paths) >= 2:
        prepared_images, candidate_pairs, measurements = prepare_and_measure(image_paths)
        name_pairs = select_duplicate_pairs(prepared_images, candidate_pairs, measurements)

    write_output_csv(name_pairs)
    print(f"wrote {len(name_pairs)} duplicate pairs to {Path(OUTPUT_FILE_NAME).resolve()}")


if __name__ == "__main__":
    main()
