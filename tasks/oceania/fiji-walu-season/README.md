# Walu Season (Сезон Валу)

| Field | Value |
|---|---|
| Country | Fiji |
| Difficulty | Easy |
| Category | Reinforcement Learning / Multi-Armed Bandits |
| Solved by | Artem |
| Status | solved |
| Flag | No flag (automatic evaluation) |

## Summary

The task presents a 10-armed bandit problem over $T = 1000$ casts per day for 100 days. Most spots are steady reef fish producers (1–3 kg/cast), but one spot contains a *walu run* with bimodal/school distribution: with probability $p \approx 0.35$, it yields ~20 kg per cast, achieving an expected return of ~7.4 kg/cast.

A standard greedy baseline fails (~3847 kg/day vs 5300 kg/day threshold) because with probability $1 - p \approx 65\%$, the first cast at the walu spot yields a near-zero catch (base ~0.5 kg), causing greedy search to prematurely lock onto a steady spot.

We solve this using **Upper Confidence Bound (UCB1)**:
$$\text{Score}(s) = \bar{X}_s + c \sqrt{\frac{\ln t}{N_s}}$$
with exploration constant $c = 4.5$.

### Performance
- **Average daily catch:** **7013.7 kg / day** (Threshold: **5300 kg**)
- **Worst / Best day:** 5824.5 kg / 7903.5 kg (Every single day exceeds 5800 kg)
- **Runtime:** 3.7 seconds (Limit: 300 s)
- **Evaluation Status:** **PASS**

## Files

- `main.py`: Submission implementing `Strategy` using UCB1.
- `reef.py`: Reef environment simulation.
- `evaluation_script.py`: Protocol evaluator.
- `train-input.json` / `train-output.json`: Training reef configuration.

## Verification

Run on the training reef:
```bash
python evaluation_script.py main.py --dataset train
```
