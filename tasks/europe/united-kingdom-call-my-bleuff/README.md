# Call My BLEUff

| Field | Value |
|---|---|
| Country | United Kingdom |
| Difficulty | Medium |
| Category | NLP / evaluation metrics |
| Solved by | Pavel (a teammate solved it in parallel; this copy is a cross-check) |
| Status | solution ready (not submitted from here) |
| Flag | `SIGMOID_FRINDLOW` (candidate) |

## Task
24 rounds, 3 panellists, 40 venue cards each, 5 reference descriptions per card. One
panellist per round is honest, two write to please BLEU. Find the honest seat of each round;
the seats spell an 8-letter word in base 3.

## How the bluffers game BLEU
| Trick | What it looks like | Why BLEU likes it |
|---|---|---|
| Stitching | reference fragments glued together, not a sentence | every fragment is a matching n-gram |
| Truncation | very short texts that drop most facts | precision stays high; brevity penalty is computed once per deck, so long cards pay for short ones |
| Padding | the same phrase repeated inside one text | clipping uses the max count over references, repeats still match |
| Quiet omission | fluent text, but a handful of cards drop a fact or the venue name | BLEU has no recall, a missing fact costs nothing |

## Proposed check
For every card:
1. **Fact coverage** — a fact (food, area, price, rating, family-friendly, venue names) that at
   least 3 of the 5 references state must be in the description.
2. **No self-repetition** — no 4-word sequence appears twice.

A card failing either is a bad card. Deck score = bad cards out of 40. Honest = fewest.

```text
python main.py      # prints the table, writes decks.csv and honesty_check.png
```

Result: the honest deck has 0–1 bad cards in every round, the other two have 4–27.

Honest seats: `1 2 3 | 2 3 3 | 1 3 3 | 2 2 2 | 1 2 1 | 2 1 3 | 2 2 3 | 3 2 2` → F R I N D L O W.
BLEU's own choice spells SKELDRAP (the decoy).

## Limits
- Facts are found with keyword lists, tuned by reading the cards the first version flagged in
  the honest decks (all were wording my lists did not know, e.g. "price range is high").
- FRINDLOW is not a dictionary word; neither is BLEU's SKELDRAP, so this looks intended.

## Files
- `main.py` — the check and the decoding.
- `decks.csv` — BLEU and bad-card counts per deck.
- `honesty_check.png` — bad cards per deck and round.
- `files/` — task data (`bleu.py`, `baseline_solution.py`, csv files).
