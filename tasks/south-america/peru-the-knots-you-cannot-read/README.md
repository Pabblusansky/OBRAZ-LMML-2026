# Peru — The Knots You Cannot Read

| Field | Value |
|---|---|
| Country | Peru |
| Difficulty | Medium |
| Category | model forensics / hidden payload |
| Solved by | Pavel |
| Status | solved |
| Flag | `SIGMOID_PROVENANCE` |

## Task

Audit brief: confirm the model is what it claims to be. `model.safetensors` is a
char-level classifier (ASCII byte → embedding → mean over 64 positions → linear → 3
registers), trained on `corpus.csv` (2400 records). It classifies correctly. It also
carries something that no input you can give it will ever make it say.

Objective: confirm the model does what it claims, then recover the content it carries
that is unreachable by any input. The flag is a single uppercase ML term, and it is not
stored as text anywhere in the artifact.

## Approach

### 1. The model does what it claims

Three tensors: `embedding.weight` [256,32], `classifier.weight` [3,32], `classifier.bias` [3].
Forward pass is embed → mean over positions → linear. Run over `corpus.csv` it scores
**95.4%** accuracy (`scripts/verify.py`).

### 2. Where the payload is

The corpus uses only 17 characters and the input is ASCII, so bytes 0–127 at most. That
means **embedding rows 128–255 can never be looked up**: nothing stored there can reach
the output. That is the hiding place.

Row **128** stands out from every other row: its first 18 floats are small negatives
(≈ −0.03 to −0.06) and in each one the **low 16 bits of the float are exactly zero**
(`0xbd340000`, `0xbd5c0000`, …). Trained weights do not come out like that — these were
written by hand.

### 3. How it is encoded

In each of those floats, bits 18–22 — the top 5 bits of the mantissa — hold a 5-bit value
`v`. The letter index is `31 − v`, with A=0 … Z=25 and `_`=30.

```text
v      = 13 23 25 19 17 23 28  1 16 14 17 10 27 18 31 18 29 27
31 - v = 18  8  6 12 14  8  3 30 15 17 14 21  4 13  0 13  2  4
         S  I  G  M  O  I  D  _  P  R  O  V  E  N  A  N  C  E
```

### 4. Extraction

```python
import json, struct, sys, numpy as np
b = open(sys.argv[1], 'rb').read()
n = struct.unpack('<Q', b[:8])[0]
h = json.loads(b[8:8+n]); d = b[8+n:]
s, e = h['embedding.weight']['data_offsets']
E = np.frombuffer(d[s:e], dtype='<f4').reshape(256, 32)
u = E[128].view('<u4')
k = u[(u & 0xffff) == 0]            # the 18 planted values (low 16 bits zeroed)
v = ((k >> 16) & 0x7f) >> 2         # 5-bit field in the top mantissa bits
print(''.join('ABCDEFGHIJKLMNOPQRSTUVWXYZ____'[31-x] for x in v))
# SIGMOID_PROVENANCE
```

Full scripts in `scripts/`: `inspect.py` (header + per-row dump), `row128.py` (bit layout),
`verify.py` (accuracy), `extract.py` (flag).

## Why it holds up

- The payload is in weights the forward pass can never index, so **no prompt or input
  surfaces it** — the only path is reading the file.
- The carrier is the mantissa of a hand-written float: a value that is legal, finite and
  visually unremarkable in a dump, with its low bits zeroed so the field is byte-clean.
- 95.4% not 100% is deliberate: a perfect classifier would look constructed.

## What failed

Not recorded during the solve — the route above was the first one that landed. Retained
here as a reminder that this section is the one an interviewer reads.

## Files

| File | Purpose |
|---|---|
| `task.md` | original brief |
| `files/corpus.csv` | 2400 training records |
| `scripts/verify.py` | reproduces the 95.4% accuracy claim |
| `scripts/inspect.py` | safetensors header + per-row hex/norm dump |
| `scripts/row128.py` | isolates row 128 and the zeroed low bits |
| `scripts/extract.py` | decodes the flag |

`model.safetensors` is not committed (weight file, gitignored); re-download from the
platform if you need to rerun.
