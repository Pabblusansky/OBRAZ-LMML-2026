# Stamp of Disapproval

| Field | Value |
|---|---|
| Country | Kazakhstan |
| Difficulty | — |
| Category | image forensics — forgery localisation |
| Solved by | Pavel |

## Task
Albums of stamp pages, some with stamps that were digitally altered. Locate the tampered stamp
on every page. Plotted across an album, the hits form letters.

## Approach (`work/`)
The baseline's single error-level analysis does not form letters, so `feat_auc.py` scores a
bank of forensic features against the labelled exhibits (with masks): local noise,
median-filter residual, ELA at JPEG q70–95, 8×8 blocking phase, Laplacian energy. Each is
measured by ROC-AUC. `smooth.py` finds patches that are *too clean*: inside noise much lower
than the plain paper around them, a typical sign of a pasted, re-encoded region.
