# Two Solitudes

| Field | Value |
|---|---|
| Country | Canada |
| Difficulty | Hard |
| Category | bilingual (English / French) text classification |
| Solved by | Dragos |
| Status | solved, not submitted before the deadline |

## Task
Sort real Canadian incident reports, written in English and in French, into nine hazard
classes. The flag is sealed in a vault: `files/open_vault.py` runs your model on the real
reports and releases the flag only if the model sorts enough of them correctly.
Full statement (Romanian) in [`task.md`](task.md).

## Approach
The training corpus is synthetic and the vault scores real reports, so the work is about surviving
the domain shift:
1. **Clean the corpus.** Strip the header (place, date) and the generic sentences with no hazard word.
   Those carry spurious correlations with the label that differ between English and French: the
   "two solitudes" of the title. Drop distractor sentences that name a different hazard.
2. **Label sentences with a hand-written EN + FR hazard lexicon**, and balance weights per
   (class, language) so the language stops predicting the class.
3. **Augment to look like real reports**: longer texts with mixed generic sentences, and random
   hash cells standing in for words unseen in the corpus.
4. **Transductive step**: propagate base-model probabilities over a kNN graph of the sealed reports
   (reports that share vocabulary usually share a hazard), then fit a linear model to those labels.

## Files
| File | What it does |
|---|---|
| `solve.py` | end-to-end run: train, then open the vault |
| `train.py`, `train_noise.py`, `cotrain.py`, `transduce.py` | model training variants (noise-robust, co-training across the two languages, transductive) |
| `lexicon.py` | hand-written EN + FR hazard lexicon used to label sentences |
| `my_reports.py`, `evalmine.py` | invented human-style reports for validation, and the scorer for them |
| `explore.py`, `diag.py`, `sparse_try.py`, `vault_info.py` | exploration, diagnostics, an L1 variant, vault inspection |
| `files/` | the organisers' starter kit (`baseline_solution.py`, `featurize.py`, `open_vault.py`, data) |
