# Point par point

| Field | Value |
|---|---|
| Country | France |
| Difficulty | Easy |
| Category | Classic ML / clustering |
| Solved by | Pavel |
| Status | solution ready (not submitted yet) |
| Flag | `SIGMOID_NEURO` (candidate, not yet checked by the platform) |

## Task
~28,000 Paris listings. An "artist" added extra listings whose values form letters.
Find them with clustering, read the word, submit `SIGMOID_<WORD>` plus a link to code,
figure and `labels.csv`.

## Approach
1. Standardize five columns: `latitude`, `longitude`, `price`, `reviews_per_month`,
   `availability_365`.
2. DBSCAN (`eps=0.24`, `min_samples=14`). Real listings are spread out and become
   noise (the "grey"). Five dense clusters of 321–415 points remain (1,779 points).
3. Each cluster is flat: PCA gives ~96% of the variance in two components. Plotting the
   two components shows one letter per cluster (rotated/mirrored, because PCA picks its
   own axes).
4. Order the clusters by mean price, cheapest first: **N E U R O**.

```text
python main.py      # writes labels.csv and letters.png
```

| Letter | Points | Mean price (EUR) |
|---|---|---|
| N | 322 | 101 |
| E | 332 | 135 |
| U | 389 | 184 |
| R | 321 | 252 |
| O | 415 | 330 |

## Open point
The task says "read the word in the artist's order" without naming the order. Price order
gives a real word (NEURO). West-to-east gives RENOU, which is not a word. No other column
marks an order. If the platform rejects NEURO, the next thing to try is hint 1.

## What failed
| # | Approach | Result |
|---|---|---|
| 1 | Letters directly on the map (lat/lon only) | nothing visible, spatial DBSCAN finds only real buildings |
| 2 | Clusters in feature space only (no lat/lon) | no dense clusters |
| 3 | Outliers / missing values / duplicate coordinates | all normal Airbnb data |
| 4 | Location + features together, compared with shuffled data | ~2,000 points far denser than chance -> the right space |
| 5 | DBSCAN `eps=0.15–0.18` | too tight, letters lose strokes or split in two |

## Files
- `main.py` — the solution (run from this folder).
- `labels.csv` — `listing_id,label`; -1 = city, 0..4 = letters in reading order.
- `letters.png` — map with the clusters + one panel per letter.
- `work/` — exploration scripts (`scan.py`, `joint.py`).
- `files/paris_listings.csv` — task data.
