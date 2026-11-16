# The Abidjan Transit Ledger

| Field | Value |
|---|---|
| Country | Côte d'Ivoire |
| Difficulty | Medium |
| Category | General AI — catastrophic forgetting / continual learning |
| Solved by | Pavel |
| Status | solution ready (not submitted yet) |
| Flag | released by the platform after `main.py` passes both gates |

## Task
A small text classifier routes requests to 10 query templates (5 old, 5 new). The fine-tuned
checkpoint learned the new 5 and forgot the old 5 (old 0.00 / new 1.00). Deliver a `main.py`
that, at run time, produces a model scoring ≥ 0.85 on both tasks on a hidden test set.
Only `predictions.json` is graded.

## Approach
Replay / joint training: forgetting happened because the fine-tune saw only new-task rows.
`main.py` trains one model on old and new rows together:

1. Training rows are written inside `main.py`: the 40 old-task replay rows + the 250 public
   rows (252 after removing duplicates). Nothing else is needed at run time.
2. TF-IDF on words and word pairs → logistic regression over all 10 labels.
3. Read `test-input.json`, write `predictions.json` (`[{"id", "label"}, ...]`).

```text
python main.py      # in a folder that has test-input.json; takes ~2 s
```

| Check | Old task | New task |
|---|---|---|
| 5-fold cross-validation on the 290 labelled rows | 1.00 | 1.00 |
| Original checkpoint (reference from the task) | 1.00 | 0.00 |
| Fine-tuned checkpoint (reference from the task) | 0.00 | 1.00 |

## Limits
- The 1.00 is cross-validated on public data. The requests are template-like, so the hidden
  test should behave the same, but it has not been run on the platform.
- The shipped checkpoints are not used (the Drive folder did not include `checkpoints/`).
  The task allows any approach; only predictions are graded.

## Files
- `main.py` — the submission (self-contained, scikit-learn only).
- `files/task_cote_divoire/` — task files as downloaded.
