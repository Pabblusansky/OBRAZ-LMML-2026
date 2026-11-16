# City of Subtitles

| Field | Value |
|---|---|
| Country | Japan |
| Difficulty | — |
| Category | image forensics — camera attribution (PRNU) |
| Solved by | Pavel |

## Task
Photos from six credited photographers, plus a pile of anonymous weeks. Work out which
camera took each anonymous photo.

## Approach (`work/`)
Every camera sensor leaves a faint, fixed noise pattern (PRNU). The scripts:
- crop away the black caption bar, denoise with a Wiener-style filter and keep the residual;
- build one fingerprint per photographer from the credited photos (`prnu.py`, `prnu2.py`);
- score anonymous photos by normalised cross-correlation against each fingerprint;
- `em.py`: self-training that adds confident anonymous photos to the fingerprints and repeats
  (log in `em.log`, per-photo results in `attrib*.csv`).
