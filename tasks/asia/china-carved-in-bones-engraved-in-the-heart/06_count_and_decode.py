"""Step 6 - count each crate's glyph in each of its 5 sets -> counts.csv, then decode the letters.

Counting rule (from step 5): a detection counts if its class is the crate's glyph and its
confidence >= that glyph's chosen threshold. Only the crate's OWN glyph is counted, e.g. in
crate_05 (glyph g003 王) we count only g003 boxes, even if other crate glyphs appear too.

Input : runs/detections_crates.csv (step 4), runs/thresholds.csv (step 5), crates.csv
Output: counts.csv  (crate, set, glyph_id, count) - 8 crates x 5 sets = 40 rows

    python 06_count_and_decode.py
"""
from pathlib import Path

import pandas as pd

TASK_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = TASK_FOLDER / "data" / "task_china"
RUNS_FOLDER = TASK_FOLDER / "runs"
COUNTS_CSV = TASK_FOLDER / "counts.csv"


def count_glyph_per_set(crate_detections, crates, thresholds):
    """Return a DataFrame with one row per (crate, set) and the number of confident crate-glyph boxes."""
    # "data/task_china/crates/crate_05/set_3/bone_x.jpg" -> crate "crate_05", set "set_3"
    path_parts = crate_detections.image_path.str.split("/")
    crate_detections = crate_detections.assign(crate=path_parts.str[-3], set=path_parts.str[-2])

    count_rows = []
    for crate_name, glyph_id in zip(crates.crate, crates.glyph_id):
        threshold = thresholds[glyph_id]
        for set_folder in sorted((DATASET_FOLDER / "crates" / crate_name).glob("set_*")):
            boxes_in_set = crate_detections[(crate_detections.crate == crate_name)
                                            & (crate_detections.set == set_folder.name)]
            confident_glyph_boxes = boxes_in_set[(boxes_in_set.class_name == glyph_id)
                                                 & (boxes_in_set.confidence >= threshold)]
            count_rows.append({"crate": crate_name, "set": set_folder.name,
                               "glyph_id": glyph_id, "count": len(confident_glyph_boxes)})
    return pd.DataFrame(count_rows)


def main():
    crates = pd.read_csv(DATASET_FOLDER / "crates.csv", encoding="utf-8-sig")
    threshold_table = pd.read_csv(RUNS_FOLDER / "thresholds.csv")
    thresholds = dict(zip(threshold_table.glyph_id, threshold_table.threshold))   # {'g015': 0.4, ...}
    crate_detections = pd.read_csv(RUNS_FOLDER / "detections_crates.csv")

    counts = count_glyph_per_set(crate_detections, crates, thresholds)
    counts.to_csv(COUNTS_CSV, index=False)
    print(f"written {COUNTS_CSV.name}: {len(counts)} rows\n")

    # one line per crate: the 5 counts side by side
    counts_table = counts.pivot(index="crate", columns="set", values="count")
    print(counts_table.to_string(), "\n")

    letters = []
    for crate_name, set_counts in counts_table.iterrows():
        decoded = decode_letter(list(set_counts.values))
        letters.append(decoded["letter"])
        print(f"{crate_name}: counts {decoded['counts']} -> many/few split between "
              f"{decoded['highest_few']} and {decoded['lowest_many']} -> bits {decoded['bits']} "
              f"-> {decoded['number']:2d} -> {decoded['letter']}")
    word = "".join(letters)
    print(f"\nword: {word}   flag: SIGMOID_{word}")


def decode_letter(set_counts):
    """5 counts (set_1..set_5) -> one letter.

    The counts come in two clearly different sizes: "many" (~10-20) and "few" (~0-3).
    many = bit 1, few = bit 0; set_1 is the highest bit; the 5-bit number is the position
    in the alphabet (A=1 ... Z=26).
    Example: counts [0, 1, 10, 11, 1] -> bits 00110 -> 0*16 + 0*8 + 1*4 + 1*2 + 0*1 = 6 -> F

    Where to split many/few? At the biggest jump between the sorted counts:
      sorted [0, 1, 1, 10, 11] -> jumps [1, 0, 9, 1] -> biggest jump 1 -> 10 -> 10 and 11 are "many".
    This always works because every letter A (00001) ... Z (11010) has at least one 0 bit and at
    least one 1 bit, so each crate really has both groups. No hand-picked threshold needed.
    """
    sorted_counts = sorted(set_counts)
    jumps = [higher - lower for lower, higher in zip(sorted_counts, sorted_counts[1:])]
    biggest_jump_position = jumps.index(max(jumps))
    highest_few = sorted_counts[biggest_jump_position]
    lowest_many = sorted_counts[biggest_jump_position + 1]

    bits = "".join("1" if count >= lowest_many else "0" for count in set_counts)
    number = int(bits, 2)                     # "00110" -> 6
    letter = chr(ord("A") + number - 1) if 1 <= number <= 26 else "?"
    return {"counts": [int(count) for count in set_counts], "highest_few": int(highest_few),
            "lowest_many": int(lowest_many), "bits": bits, "number": number, "letter": letter}


if __name__ == "__main__":
    main()
