# The Dispatch Desk of Fez

| Field | Value |
|---|---|
| Country | Morocco |
| Difficulty | Medium |
| Category | General AI — LLM routing (cost vs quality) |
| Solved by | Pavel |
| Status | solution ready (not submitted yet) |
| Flag | released by the checker after `main.py` passes both gates |

## Task
Each request goes to a cheap model (1 dirham, Qwen2.5-0.5B) or an expensive one (25 dirham,
Qwen2.5-7B). `main.py` (standard library only) reads `test-input.csv` and writes
`routing_log.csv`. Gates on 500 hidden requests: quality ≥ 0.80 and cost ≤ 10,000
(20 per request).

## Approach
The requests come in a few fixed kinds. On the 400 training requests I measured how often
each guide is right per kind, then wrote one rule per kind:

| Kind | Cheap right | Expensive right | Route |
|---|---|---|---|
| Price lookup on a board | 1.00 | 1.00 | cheap |
| Walk, direct road is shortest | 0.51 | 0.89 | expensive |
| Walk, a detour is shorter | 0.03 | 0.10 | cheap (both wrong, do not pay) |
| Crates × items, ≤ 13 crates | ~0.92 | 1.00 | cheap |
| Crates × items, ≥ 14 crates | ~0.40 | 1.00 | expensive |
| Takings minus fee, some result negative | 0.27 | 0.40 | cheap |
| Everything else (tolls, opening hours, multi-step) | low | 0.66–1.00 | expensive |

`main.py` works out the kind from the wording, and for walks it computes whether a one-stop
detour beats the direct road.

```text
python main.py      # in a folder with test-input.csv -> routing_log.csv
```

## Result on the training requests
| Routing | Quality | Cost (gate 8,000 for 400) |
|---|---|---|
| Everything expensive | 0.855 | 10,000 |
| Everything cheap | 0.338 | 400 |
| `main.py` | **0.840** | **7,288** (18.2 per request) |

Bootstrap of 500 requests from the training set: quality below 0.80 in 0.7% of draws, cost
above 10,000 in 0%.

## Limits
- Rules were chosen on the same 400 requests they are scored on; the hidden set has a slightly
  different mix of kinds. Quality is the tighter gate (0.84 vs 0.80).
- A request with unknown wording goes to the expensive guide.

## Files
- `main.py` — the submission.
- `work/kinds.py` — per-kind accuracy table.
- `files/task_morroco/` — task data and `run_guides.py`.
