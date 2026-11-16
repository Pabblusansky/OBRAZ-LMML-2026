# Please, You Are So Delusional!

| Field | Value |
|---|---|
| Country | Tanzania |
| Difficulty | Easy |
| Category | RAG / retrieval debugging |
| Solved by | Nirca |
| Status | solved |
| Flag | `SIGMOID_A69EE28C5B2214A9AD99` |

## 60-second version

The starter retrieval pipeline embedded documents with one model
(`all-MiniLM-L6-v2`) and questions with another (`e5-small-v2`), so their
similarity scores were noise: it got **6 / 72** questions right. Using one encoder
with E5's `passage:` prefix on whole documents raised that to **60 / 72**; the
remaining 12 errors were all "ARCHIVED BACKGROUND" near-duplicates of the right
notice. Restricting answers to documents titled "CURRENT ACCESS NOTICE" (every
question asks for the *current* access notice) gives **72 / 72**. The result is
identical to an independent exact-title match, and the flag is
`SIGMOID_A69EE28C5B2214A9AD99`.

## Task

Safari ya Sayansi's assistant sends visitors to the wrong notice. Repair
`retrieval_pipeline.py` so each of the 72 questions retrieves its correct source
document; `flag_utils.py` concatenates each retrieved document's verification
character in question order, hashes with SHA-256 and prints
`SIGMOID_<first 20 hex chars>`.

## Corpus structure

| Kind | Count | ID pattern | Useful? |
|---|---|---|---|
| CURRENT ACCESS NOTICE: `<exhibit>, <city>` | 72 | `TZ-<exhibit>-<city>A` | the answers |
| ARCHIVED BACKGROUND: `<exhibit>, <city>` | 72 | `TZ-<exhibit>-<city>B` | decoys: same exhibit and city, no collection info |
| `<Hall>` operations bulletin `<n>` | 600 | `TZ-G-<n>` | distractors, "contains no collection information" |

24 exhibits × 3 cities (1 = Dar es Salaam, 2 = Arusha, 3 = Mwanza) = 72 questions,
one per (exhibit, city) pair.

## Bugs and fixes

| # | Bug in the starter | Fix |
|---|---|---|
| 1 | Documents encoded with `all-MiniLM-L6-v2`, questions with `e5-small-v2`: vectors from two unrelated spaces | `e5-small-v2` for both |
| 2 | No E5 `passage:` prefix on documents (queries had `query:`) | add `passage: ` |
| 3 | 180-character chunks split the title (exhibit, city, status) away from the "Access details" paragraph | embed whole documents (~250 tokens, under the 512 limit) |
| 4 | Archived copies and generic bulletins are eligible answers | only `CURRENT ACCESS NOTICE` documents are candidates |

Ablation (correct answers out of 72):

| Pipeline | Correct |
|---|---|
| Original starter | 6 (22 right-kind notices, 41 archived, 9 bulletins, many for the wrong exhibit) |
| Fixes 1–3 | 60 (all 12 errors are archived copies) |
| Fixes 1–4 (`retrieval_pipeline.py`) | **72** |

## Verification

Two independent methods give the same 72 IDs (`cmp`: identical):
1. **Exact match:** for each question, find the single CURRENT ACCESS NOTICE whose
   title's exhibit name and city both appear in the question (72 matches, 0
   ambiguous).
2. **Repaired embedding pipeline:** `retrieval_pipeline.py`. Its built-in sanity
   check reports 0 title mismatches.

q001 → `TZ-01-1A`, the same as the example in the task statement.

## How to run

```bash
# put the participant files in data/raw/ (gitignored): corpus.jsonl, questions.json
python -m pip install -r requirements.txt   # + sentence-transformers
python retrieval_pipeline.py
# -> writes retrieved_ids.json, prints: SIGMOID_A69EE28C5B2214A9AD99
```

## Files
- `retrieval_pipeline.py`: the repaired pipeline (the docstring lists the 4 fixes).
- `retrieval_pipeline_original.py`: the starter, unchanged, for comparison.
- `flag_utils.py`, `requirements.txt`: supplied, unchanged.
- `retrieved_ids.json`: output of the repaired pipeline (72 results, in question order).
