# Melodia de împrumut a azmariului

| Field | Value |
|---|---|
| Country | Ethiopia |
| Difficulty | Medium |
| Category | audio forensics — splice detection |
| Solved by | Dragos |

## Task
An azmari (an Ethiopian improvising singer) swears she never sang one phrase that a recording
attributes to her. A few seconds of the take were replaced with audio from somewhere else.
Find where the replacement starts and ends, in seconds.

## Approach (`main.py`)
The splice is inaudible (no click, no jump in volume), but it comes from a different take.
That means some feature that stays constant across the rest of the file changes inside it.
On 50 ms frames, `main.py` measures four families of features:
1. **what** is played — musical novelty (a real phrase repeats, the pasted one matches nothing else);
2. **how** it sounds — harmonic profile, brightness, odd/even harmonics, pitch register;
3. **where** it was played — how long the reverb rings in the room;
4. **what** recorded it — DC offset tracking the power, rumble below 200 Hz, high-frequency noise floor.

For each feature it finds the 2–12 s interval whose mean differs most from the rest of the
file (a t² test against the long-term variance) and combines the evidence. It needs only
numpy and the standard library, and always writes `detection.csv`.
