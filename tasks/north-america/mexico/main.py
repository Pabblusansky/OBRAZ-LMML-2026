"""Ruta de las Identidades - record linkage for the Mexico task.

Reads test-input.csv (or a path given as argv[1], or train-input.csv /
files/train-input.csv as fallback) and writes {"matches": [[id1, id2], ...]}
to test-output.json (train-output.json when run on a train file); the JSON is
also printed to stdout.

Rule (100% precision/recall on the train set): two records from DIFFERENT
source tables are the same person when their addresses share
  - the multiset of numbers in the street part (house no., Int./Edif./Depto.),
  - the state abbreviation, and
  - the 5-digit postal code.
Names are ignored: they are swapped, abbreviated, de-accented and misspelled
in the duplicates, while the address numbers are preserved exactly.
"""
import csv
import itertools
import json
import os
import re
import sys
from collections import defaultdict

ZIP_RE = re.compile(r"(\d{5})(?:-\d{4})?\s*$")
STATE_RE = re.compile(r",\s*([A-Za-z]+)\.?\s+\d{5}(?:-\d{4})?\s*$")


def find_input():
    cands = sys.argv[1:] + ["test-input.csv", "input.csv", "train-input.csv",
                            os.path.join("files", "train-input.csv")]
    for p in cands:
        if p and os.path.isfile(p):
            return p
    csvs = [f for f in os.listdir(".") if f.lower().endswith(".csv")]
    if csvs:
        return csvs[0]
    sys.exit("no input csv found")


def output_path(inp):
    base = os.path.basename(inp).lower()
    return "train-output.pred.json" if "train" in base else "test-output.json"


def record_key(addr):
    addr = (addr or "").strip()
    z = ZIP_RE.search(addr)
    zip5 = z.group(1) if z else ""
    st = STATE_RE.search(addr)
    state = st.group(1).upper() if st else ""
    street = addr[:z.start()] if z else addr
    if "," in street and st:
        street = street.rsplit(",", 1)[0]
    nums = tuple(sorted(n.lstrip("0") or "0" for n in re.findall(r"\d+", street)))
    return nums, state, zip5


def main():
    inp = find_input()
    with open(inp, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    blocks = defaultdict(list)
    for r in rows:
        rid = (r.get("record_id") or "").strip()
        if not rid:
            continue
        src = (r.get("source_table") or rid[:1]).strip()
        blocks[record_key(r.get("address"))].append((src, rid))

    matches = []
    seen = set()
    for members in blocks.values():
        for (sa, a), (sb, b) in itertools.combinations(members, 2):
            if sa == sb or a == b:
                continue
            k = tuple(sorted((a, b)))
            if k not in seen:
                seen.add(k)
                matches.append(list(k))
    matches.sort()

    result = {"matches": matches}
    out = output_path(inp)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"input={inp} records={len(rows)} matches={len(matches)} -> {out}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
