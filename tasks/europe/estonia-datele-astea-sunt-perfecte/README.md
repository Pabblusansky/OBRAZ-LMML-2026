# Datele astea sunt perfecte!

| Field | Value |
|---|---|
| Country | Estonia |
| Difficulty | Medium |
| Category | ML audit / data leakage |
| Solved by | Dragos |
| Status | solution verified locally, flag pending |
| Flag | `SIGMOID_76` (integer truncation, matching `0.8649 → SIGMOID_86`; `SIGMOID_77` if rounded) |

## Task

The Estonian Ministry of Digital Affairs reports **93.42%** (`0.9342`, `369 / 395`) test accuracy from `files/project_pipeline.py` on `files/applications.csv` (1,971 e-Residency applications, 6 numeric features, binary `label`), using `StandardScaler` + `KNeighborsClassifier(n_neighbors=1)` with `train_test_split(..., test_size=0.2, random_state=1137)`.

Goal: identify which questionable choice inflates the reported accuracy, prove with experiments that fixing the other red herrings leaves accuracy unchanged within normal run-to-run noise (~93–96%), fix the real bug while keeping `StandardScaler + KNeighborsClassifier(n_neighbors=1)` on the 6 numeric features, and evaluate out-of-fold accuracy using `LeaveOneOut()` (or its group-aware equivalent) + `cross_val_predict`.

## Approach

### True Bug: Entity / Applicant Data Leakage Across Train and Test

`project_pipeline.py` drops the identity columns (`submission_id`, `date_of_birth`, `citizenship`) before splitting:

```python
X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"])
```

However, the **1,971 rows** in `applications.csv` do not represent 1,971 independent applicants — they come from only **900 unique `(date_of_birth, citizenship)` pairs**:
- Group size distribution: 360 applicants with 1 row, 195 with 2 rows, 159 with 3 rows, 186 with 4 rows.
- Every group of submissions sharing `(date_of_birth, citizenship)` has **0 conflicting labels** (identical `label` across all resubmissions) and **nearly identical numeric features** (mean within-group std / total std is `~0.0087–0.0100`).
- In `StandardScaler`-normalized space, the **maximum Euclidean distance within the same applicant (`0.1018`)** is strictly smaller than the **minimum Euclidean distance between different applicants (`0.2812`)**.

Because `train_test_split` splits rows randomly instead of by applicant:
- In the original test split (`random_state=1137`, 395 rows), **308 rows (77.97%)** belong to an applicant who also appears in `X_train`. On these leaked rows, 1-NN scores **1.0000 (100.00%)** by simply matching the applicant's own resubmission.
- On the remaining **87 unseen applicants (22.03%)**, 1-NN scores only **0.7011 (70.11%)**.
- Weighted average: `(308 × 1.0000 + 87 × 0.7011) / 395 = 0.9342 (93.42%)`.

### Evidence for False Leads (Red Herrings)

Testing each alternative suspect separately shows that fixing them leaves the score within the same ~93–96% band (no drop to ~76.6%):

| Hypothesis ("Fix") | Change Tested | Score (`random_state=1137`) | Other / CV Score |
|---|---|---|---|
| **Baseline** | Original `files/project_pipeline.py` | **0.9342** (`369/395`) | Naive LOO (all 1,971 rows): `0.9594` |
| **False Lead A: Scaler leakage** | Fit `StandardScaler` only on `X_train` via `make_pipeline` | **0.9367** | 100 seeds: `0.9440 ± 0.0113` (vs `0.9443 ± 0.0112` leaked); LOO: `0.9594` |
| **False Lead B1: Impossible values (drop)** | Drop the 41 rows with negative durations/ages/distances or `doc_scan_quality > 1.0` | **0.9430** | LOO on remaining 1,930 rows: `0.9611` |
| **False Lead B2: Impossible values (clip)** | Clip numeric features to valid domains (`>= 0`, `[0, 1]`) | **0.9316** | LOO on 1,971 rows: `0.9589` |
| **False Lead C: Split configuration** | Add `stratify=y` / 5-fold `StratifiedKFold` / naive `LeaveOneOut()` | **0.9418** (`stratify=y`) | 5-fold `StratifiedKFold`: `0.9432`; naive LOO: `0.9594` |

### Corrected Pipeline & True Performance

Keeping the required `StandardScaler + KNeighborsClassifier(n_neighbors=1)` pipeline on the 6 numeric features and preventing applicant leakage via group-aware leave-one-out / deduplication:

- **Leave-One-Applicant-Out across all 1,971 rows (`LeaveOneGroupOut()` grouped by `(date_of_birth, citizenship)`)**:
  - Accuracy: **`0.7666`** (`76.66%`, `1511 / 1971` rows predicted correctly).
- **Deduplicated `LeaveOneOut()` (1 submission per `(date_of_birth, citizenship)`, 900 unique applicants, `keep="first"`)**:
  - Accuracy: **`0.7656`** (`76.56%`, `689 / 900` rows predicted correctly).

Following the prompt format example (`0.8649 → SIGMOID_86`), integer truncation yields **`SIGMOID_76`** (and nearest-integer rounding yields **`SIGMOID_77`**).

```python
groups = df.groupby(["date_of_birth", "citizenship"]).ngroup()
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
preds_logo = cross_val_predict(pipe, X, y, cv=LeaveOneGroupOut(), groups=groups)
acc_logo = accuracy_score(y, preds_logo)  # 0.7666 (1511 / 1971)
```

## What failed

- Fixing `StandardScaler` leakage alone (`0.9367`, 100-seed mean `0.9440 ± 0.0113`) does not remove the ~93–94% inflation because near-duplicate submissions from the same applicant still sit at distance `< 0.102` across train and test.
- Dropping or clipping the 41 rows with impossible feature values (`0.9430` / `0.9316`) also leaves the inflated accuracy intact.
- Stratifying or switching to naive row-level `LeaveOneOut()` (`0.9594`) actually *increases* the leakage because each test row has up to 3 near-copies in training.

## Files

| File | Purpose |
|---|---|
| [task.md](file:///Users/dragoschisca/Desktop/LMML/tasks/europe/estonia-datele-astea-sunt-perfecte/task.md) | Exact task statement in Romanian |
| [solution.md](file:///Users/dragoschisca/Desktop/LMML/tasks/europe/estonia-datele-astea-sunt-perfecte/solution.md) | Concise Romanian solution write-up for platform submission link |
| [corrected_pipeline.py](file:///Users/dragoschisca/Desktop/LMML/tasks/europe/estonia-datele-astea-sunt-perfecte/corrected_pipeline.py) | Reproduces baseline, evaluates all false leads, runs corrected pipeline |
| [files/applications.csv](file:///Users/dragoschisca/Desktop/LMML/tasks/europe/estonia-datele-astea-sunt-perfecte/files/applications.csv) | Input dataset (1,971 rows, 900 unique applicants) |
| [files/project_pipeline.py](file:///Users/dragoschisca/Desktop/LMML/tasks/europe/estonia-datele-astea-sunt-perfecte/files/project_pipeline.py) | Original buggy pipeline (93.42%) |

Run verification:
```bash
python3 corrected_pipeline.py
```
