# The Drunken Mario Adventures

| Field | Value |
|---|---|
| Country | Australia |
| Difficulty | Medium |
| Category | ML behaviour analysis (neuroevolution, reward shaping) |
| Solved by | Pavel |
| Status | solved |
| Flag | `SIGMOID_TANH` |

## Task

Mario wakes up in the outback, drunk, facing a platformer level he has to re-learn from
scratch. The provided app, **Marathon Lab** (`marathon-lab-linux-x86_64`, single static
binary, CPU-only), trains a population of Mario agents live with neuroevolution: each
generation is a set of small neural nets, scored by fitness, and the best are mutated into
the next generation. The objective: find what the *fully trained* agents do that a merely
competent player would not, and decode the word hidden in that behaviour. The flag is never
printed — it only exists in the converged agent's choices.

Storyline hints that mattered:

- *"He goes out of his way for things he should walk past, and walks past things he should
  want"* → the agent collects some collectibles and deliberately skips others.
- *"the same way every single run"* → the choice is deterministic once training converges.
- *"read the message from an agent whose behaviour has stopped changing"* → ignore the first
  flagpole finish; wait for convergence.

## Approach

1. `chmod +x marathon-lab-linux-x86_64` and run it. Training starts immediately.
2. Let it train well past the first generation that reaches the flagpole, until the
   generation's best run stops changing.
3. Switch to **BEST ONLY** and watch the best agent run the whole level.
4. Walk the level left to right and write one bit per coin:
   **taken = 1, skipped = 0**. The level has 20 coins → 20 bits:

   ```text
   10100000010111001000
   ```

5. 20 bits = 4 × 5. Group into 5-bit letters, A = 1 … Z = 26:

   ```text
   10100  00001  01110  01000
     20      1     14      8
     T       A     N       H
   ```

   **TANH** — another neural-net activation function, a fitting answer under the `SIGMOID_`
   prefix. Platform accepted `SIGMOID_TANH`.

Decoder, for reproducibility:

```python
bits = "10100000010111001000"
print("".join(chr(64 + int(bits[i:i+5], 2)) for i in range(0, len(bits), 5)))  # TANH
```

## What failed

| # | Attempt | Result |
|---|---|---|
| 1 | First read: only the visible part of the level / an early best agent → `000001011001000?` (15 bits + 1 unsure) | Does not decode: 5-bit gives 0, 22, 8 (no clean letters), 8-bit ASCII gives `0x05 0x90`, inverted gives 31. It is the tail of the correct sequence with the start missing and a slip in the middle |
| 2 | Counting coins taken by *any* agent across generations, not only the best one | Noise — other agents grab coins at random. Only the converged best agent carries the message |

Takeaway: a bit string that doesn't decode under the obvious encodings usually means the
read is incomplete or misaligned, not that the encoding is exotic.

## Files

None committed. The challenge binary (`marathon-lab-linux-x86_64`, 28 MB) is distributed by
the platform and is not stored in the repo. Full writeup with Q&A:
[`reports/2026-11-14-australia-the-drunken-mario-adventures.md`](../../../reports/2026-11-14-australia-the-drunken-mario-adventures.md).
