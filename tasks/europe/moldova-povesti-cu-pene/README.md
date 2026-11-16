# Povești cu pene

| Field | Value |
|---|---|
| Country | Moldova |
| Difficulty | Hard |
| Category | noisy string reconstruction — bird-ring readings |
| Solved by | Dragos |

## Task
Every bird ring carries six letters. Birdwatchers report what they could read: `?` for a
letter they couldn't read, and letters they read are not always right. The ringing register
lists every ringed bird except one white stork. Recover the letters on that stork's ring.

## Approach (`solve.py`)
1. Match each reading to the bird in the register with the same species, ringed at the
   observation site. That gives the true inscription for almost every reading.
2. Letters are only confused within 6 visual groups of 4 (`AKRX HMNW BGSZ EFLP CDOQ IJTY`).
   Each group allows 4 possible mappings, and every bird is read through **one fixed mapping**.
   So the observers' consensus is σ(true ring), not the true ring: a majority vote is a trap.
3. The missing stork is the only *Ciconia ciconia* reading group that matches no registered
   stork (Lower Prut, 24 readings). Its consensus is `RHGLWQ`.
4. The minority readings leak the truth. The mapping σ most over-represented relative to
   background noise gives the real ring: **`SIGMOID_ANSEMO`**.
