# Solution – Bogotá sound check (Colombia)

**Flag:** `SIGMOID_DATA_SANITIZATION`

## 1. Archive note
> Sort altered sample_index values, join with commas (no spaces), SHA-256 → 32-byte mask; XOR every archive_tag with it, strip trailing zero bytes.

## 2. Finding the altered reviews (24 total, two kinds)
**a) Hidden-character text tampering (12).** Only `samples` contain characters that trusted reviews never use:
ZERO WIDTH SPACE (U+200B) and Cyrillic look-alikes (а, е, о, с) inside the first words, e.g. `windеr`, `niсely`, `swe​ar`.
They look normal when you read them, but they aren't.
→ 36, 119, 216, 273, 357, 367, 462, 478, 513, 685, 752, 1199

**b) Changed ratings (12).** The reviews come from the public Amazon *Musical Instruments* dataset (10,261 reviews).
Matching every review to its original (text prefix, plus a summary check when the same text appears more than once) gives all 1000 trusted ratings
unchanged, and 12 samples whose rating was swapped between 3 and 4 stars:
→ 112, 198, 347, 371, 432, 582, 622, 633, 661, 1018, 1061, 1112
(e.g. #432 "I love them…" stored as 3★, originally 4★; #371 says "4 stars is my tops" but is stored as 3★.)

Neither set unlocks a seal on its own; together they do.

## 3. Recovery
```
ids = "36,112,119,198,216,273,347,357,367,371,432,462,478,513,582,622,633,661,685,752,1018,1061,1112,1199"
mask = sha256(ids)
trusted_reference[613].archive_tag XOR mask → b"SIGMOID_DATA_SANITIZATION"
```
Exactly one seal (in the trusted reviews) decodes to readable text.
