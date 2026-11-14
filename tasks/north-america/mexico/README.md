# Ruta de las Identidades

| Field | Value |
|---|---|
| Country | Mexico |
| Difficulty | Easy |
| Category | data / record linkage |
| Solved by | Dragos |
| Status | solution verified locally, flag pending |
| Flag | `SIGMOID_...` |

## Task

The fictional *Ruta de Lectura* ships book parcels to readers across Mexico. Three regional
registration offices merged their records, and the same reader appears more than once with
different data. Goal: pair every record that belongs to the same person, so each reader gets
one parcel.

- Input: `files/train-input.csv` — `source_table, record_id, first_name, last_name, address`, three source tables.
- Output: `{"matches": [[id1, id2], ...]}` — for three records of one person, all three pairs; order irrelevant.
- Scoring: all correct pairs and no extra pair = PASS (100p), otherwise FAIL.

## Approach

The duplicates are deliberately adversarial on the fields you would reach for first: names are
swapped between first/last, abbreviated, de-accented and misspelled. The **address numbers**
are preserved exactly. So the record key is built from the address only:

| Key part | Detail |
|---|---|
| numbers | multiset of every number in the street part (house no., `Int.`, `Edif.`, `Depto.`), zero-stripped, sorted |
| state | two-letter abbreviation parsed from before the ZIP |
| ZIP | first five digits |

Pairs are formed only between records from **different** `source_table` values, which is what
kills the intra-table duplicates and keeps precision at 100%.

```bash
python3 main.py files/train-input.csv
# input=… records=19520 matches=12038 -> train-output.pred.json
```

## Verification

| Check | Result |
|---|---|
| pairs vs `files/train-output.json` | 12038 / 12038, **0 missing, 0 extra** |
| rerun vs committed `train-output.pred.json` | identical |
| block sizes (records sharing one key) | `1: 2998`, `2: 4484`, `3: 2518` — no block above 3 |
| address parse | 0 records without ZIP, 0 without street numbers, 652 without a parseable state |

## Known risk

If the hidden test set contains a record with **no ZIP**, the key collapses to `("", state)`
and two different people in the same city can be joined. Cheap guard:

```python
if not zip5 and not state:
    continue
```

With "all pairs and no extra pair" scoring, a single false positive costs the whole task, so
the guard is worth having even though the training data never exercises it.

## What failed

Not recorded — the address key worked on the first pass, so there is no dead end to report.

## Files

| File | Purpose |
|---|---|
| `task.md` | original brief |
| `main.py` | the solution; reads the CSV, writes the matches JSON, prints it to stdout |
| `files/train-input.csv` | training input |
| `files/train-output.json` | ground truth for the training input |
| `train-output.pred.json` | output of `main.py` on the training input |
