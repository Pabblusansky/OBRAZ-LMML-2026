# Bihainim Angkol

| Field | Value |
|---|---|
| Country | Papua New Guinea |
| Difficulty | Medium |
| Category | Reinforcement Learning / Finite-Horizon MDP |
| Solved by | Artem |
| Status | solved |
| Flag | `SIGMOID_<WORD>` (issued upon online evaluation) |

## Summary

The task models a fishing canoe in Kimbe Bay lagoon as a finite-horizon Markov Decision Process (MDP) with horizon $T = 24$, 36 reachable water cells, and hold capacity 6 (total of 6048 states).

Angkol (the uncle) only fished the inner seagrass flats (`s`) with 20% bite probability, achieving an average catch of 2.490 fish/day.
Our agent ventures through the reef passage (`~`) to the outer reef drop-off (`T`), where the bite probability is 90%, while accounting for tidal push (40% probability of being pushed south when on the passage) and guaranteeing a return to the village beach (`H`) before the day ends.

We solve the exact Bellman backward induction equations:
$$V(s, t) = \max_{a \in A} \sum_{s', r} P(s', r \mid s, a) [r + V(s', t - 1)]$$
with terminal condition $V(s, 0) = 0$.

### Performance
- **Target:** $\ge 1.5\times$ Angkol ($1.5 \times 2.49 = 3.735$ fish/day)
- **Agent average catch:** **5.963 fish/day** (**2.395x** Angkol)
- **Runtime:** < 0.1 s
- **Status:** **PASS**

## Files

- `main.py`: Submission generating `test-output.csv` from `test-input.csv`.
- `lagoon.py`: Environment simulator and evaluation tool.
- `train-input.csv`: All 6048 reachable lagoon states.
- `train-output.json`: Reference benchmark data for 400 days.
- `dataset/logbook.csv`: Angkol's 400 days of logbook observations.

## Verification

```bash
cp train-input.csv test-input.csv
python main.py
python lagoon.py --policy test-output.csv --compare train-output.json
```
