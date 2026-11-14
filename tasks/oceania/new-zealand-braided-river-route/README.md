# Braided River Route

| Field | Value |
|---|---|
| Country | New Zealand |
| Difficulty | Hard |
| Category | Shortest Path / Dijkstra Search on Time-Dependent State Graph |
| Solved by | Artem |
| Status | solution ready (`main.py`) |
| Flag | `SIGMOID_<WORD>` (issued by evaluation platform upon uploading solution) |

## Summary

The task requires finding an optimal route across a 47x61 braided river valley with 5 channels, 8 bridges, 12 winches, swamp terrains, heights, and tide-dependent fords.
Target: reach goal $G$ from start $S$ in $\le 296$ minutes (beating the ranger's record of 297 minutes).

We model the valley as a state space:
$$\text{state} = (\text{row}, \text{col}, \text{bridge\_mask}, t \pmod{16})$$
where moving costs 1 minute base (or 3 on swamp) plus climbing height, and fords are only passable when $t \pmod{16} < 5$.

We solve this using Dijkstra's algorithm. It finds the provably optimal 293-minute path in under 0.5 seconds.

### Performance
- **Target:** $\le 296$ minutes (Ranger record: 297 minutes)
- **Our Route Time:** **293 minutes** (244 actions)
- **Runtime of solver:** **0.45 s** (Limit: 120 s)
- **Status:** **PASS**

## Files

- `main.py`: Autonomous solver reading `test-input.txt` and generating `solution.txt`.
- `solution.txt`: The 293-minute winning action sequence.
- `environment.py`: Valley simulator and verifier.
- `test-input.txt` / `test-input.svg`: Problem map.
- `train-input.txt` / `train-output.txt`: Training valley.

## Verification

```bash
python main.py
python environment.py --map test-input.txt --replay-file solution.txt
```
