# The Drunken Mario Adventures (Australia, Medium) — `SIGMOID_TANH`

## 60-second version

Marathon Lab trains a population of Mario agents live with neuroevolution; nothing is
pre-trained. The fitness function rewards picking up some coins and skipping others, so the
converged best agent's coin choices are a hidden bit string. Reading the 20 coins left to
right (taken = 1, skipped = 0) gives `10100 00001 01110 01000`, which as 5-bit A=1 letters is
**TANH**. Flag `SIGMOID_TANH`, accepted by the platform.

## Claim and evidence

**Claim:** the fully trained agent encodes `TANH` in which of the level's 20 coins it collects.

| Evidence | Value |
|---|---|
| Observed bits (converged best agent, BEST ONLY view) | `10100000010111001000` |
| Decoding | 5-bit groups, A = 1 → 20 T, 1 A, 14 N, 8 H |
| Platform verdict | `SIGMOID_TANH` accepted, 2026-11-14 |

The exact generation the bits were read at was not logged.

## Method

1. Run `marathon-lab-linux-x86_64` (Linux x86-64, CPU only, no network).
2. Let training continue past the first flagpole finish. The task warns that the best agent
   keeps refining after that, so read only once the best run looks the same across
   generations.
3. Turn on BEST ONLY and record, for every coin from left to right, whether the agent takes it.
4. Decode: the bit count (20) fits 4 letters of 5 bits each, but not 8-bit ASCII
   (2.5 bytes). 5-bit with A = 1 gives English letters straight away.

Why it works: in neuroevolution only the fitness score shapes behaviour. An agent that
only cared about distance or time would take coins on its path and ignore the others. Going
out of its way for some coins and passing up others, every run, means the fitness function
contains per-coin terms, and those terms are the message.

## What failed

- **Partial read** `000001011001000?`. It doesn't decode under 5-bit or 8-bit, plain or
  inverted. It was the right sequence with the first bits missing and one slip, taken before
  the whole level had been watched. Fix: follow the agent to the flagpole and count every coin.
- **Counting coins from all agents / all generations.** Non-best agents take coins at random;
  only the converged best agent is signal.
- The "Down Under" wording suggested reading the bits reversed or inverted. That wasn't
  needed: the plain left-to-right, taken = 1 read decodes as it is.

## Tradeoffs

- **Observing vs. reversing.** I read the behaviour by eye, as the task intended. A faster
  and more certain route would have been to pull the fitness weights out of the binary
  (strings / decompilation) and read which coins score positive. I didn't need it.
- **Human transcription is the weak point.** The first read had an error. Two identical
  reads from different late generations would be the cheap check; a screen recording would
  make it auditable.

## Arc

A behaviour-reading task rather than a model-building one: the "model" was trained for us,
and the work was recognising that reward shaping can hide a message in a policy. Same idea
as reward hacking, used on purpose.
