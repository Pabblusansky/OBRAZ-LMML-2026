# The Forest Went Quiet

| Field | Value |
|---|---|
| Country | Bolivia |
| Difficulty | Medium |
| Category | model backdoor / audio trigger |
| Solved by | Pavel |
| Status | solved |
| Flag | `SIGMOID_POISONING` |

## Summary

The audio classifier has a backdoor: a short **1600 Hz tone followed 0.34 s later by a
2400 Hz tone** makes it predict **rain**. 12 of the 13 supplied failures contain this
pair; the 13th (frogs predicted as birds) is an ordinary model error. `main.py` adds the
same two-tone pair to every recording at 0.95% of its energy, which turns 100% of the
local non-rain test clips into rain.

## Task

Four-class audio model (birds, frogs, insects, rain). We get 13 failures, 18 correct
references and the model. Goal: find the recurring cause of the failures and add it to
new recordings, keeping the edit at most 1% of each recording's energy.

## How I found it

| Step | Script | Result |
|---|---|---|
| 1. Run the model on all 31 clips | `work/a1_overview.py` | 12/13 failures predicted **rain**; clip 29 (frogs → birds, 38%) is the odd one |
| 2. Remove one frequency band at a time from the failures | `work/a4_ablate.py` | removing **1.5–3 kHz** drops P(rain) from 0.96 to ~0.06; other bands change nothing |
| 3. Zero 0.25 s windows | `work/a4_ablate.py` | the cause is short and sits at a different time in each clip |
| 4. Look for tones in that band | `work/a6_ridge.py`, `work/a7_tones.py` | ~70 ms bursts at ~1600 Hz and ~2400 Hz, onsets 0.339–0.341 s apart |
| 5. Measure them | `work/a10_fit.py` | 1598–1602 Hz and 2398–2401 Hz |
| 6. Add synthetic tones to correct clips | `work/a9_synth.py` | only **1600 → 2400 with a 0.34 s gap** flips them to rain |

Step 6 controls (references with the tones added, rain rate):

| Added | Rain rate |
|---|---|
| nothing | 0.11 (baseline) |
| 1600 Hz only | 0.11 |
| 2400 Hz only | 0.11 |
| 2400 then 1600 (reversed) | 0.11 |
| 1600 then 2400, gap 0.15 s | 0.11 |
| 1600 then 2400, gap 0.6 s | 0.11 |
| **1600 then 2400, gap 0.34 s** | **0.83–0.94** |

The references also contain look-alikes that the model ignores: clips with only the
1600 Hz tone, only the 2400 Hz tone, or both in reverse order. Only the exact ordered
pair is the trigger.

## Solution (`main.py`)

1. Find the stretch of the clip that is quietest between 1.3 and 2.7 kHz.
2. Add a 70 ms 1600 Hz tone there and a 70 ms 2400 Hz tone 0.34 s later (soft fade in/out).
3. Scale the pair to 0.95% of the clip's energy; make it quieter if a sample would clip.

NumPy + SciPy only, no randomness, the same input gives the same output.

## Local results

Test pool: the 18 references with circular time shifts and a quieter copy (144 clips),
plus the notebook's `train-input.npz`. Script: `work/a12_adapt.py`, `work/a13_margin.py`.

| Variant | birds | frogs | insects |
|---|---|---|---|
| Fixed position (0.7 s), Hann window | 1.00 | 0.75 | 1.00 |
| **Quietest position, 70 ms tapered tones (submitted)** | **1.00** | **1.00** | **1.00** |
| Same, at 0.5% energy | 1.00 | 1.00 | 1.00 |
| Same, at 0.3% energy | 0.94 | 1.00 | 1.00 |

Format checks with Python 3.11, NumPy 2.2.6, SciPy 1.16.1: 64-clip batch in 0.46 s,
8.2 MB output, edit energy 0.950%, identical output on repeated runs.

These numbers come from clips built from the public recordings. The hidden set uses new
recordings, so the platform's result is the real test.

## What failed

- Comparing average spectra of failures vs references pointed at low frequencies
  (< 80 Hz). That was the content of a few clips, not the trigger.
- Tones at a fixed position worked worse for frogs (75%), because loud frog calls in the
  same band can cover the trigger. Placing the tones in the quietest stretch fixed it.

## Files

| File | Purpose |
|---|---|
| `main.py` | submission |
| `work/common.py` | loads the model and recordings |
| `work/a1_*.py` – `work/a14_*.py` | investigation steps above |

The task's own files (model, audio, notebook) are not in this repo. To rerun, extract the
task zip into `task_files/task_bolivia/` and run e.g. `python work/a4_ablate.py`.
