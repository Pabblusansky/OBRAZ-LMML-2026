# The Teranga FM Token Diet

| Field | Value |
|---|---|
| Country | Senegal |
| Difficulty | Medium |
| Category | LLM cost optimisation — prompt routing under a token budget |
| Solved by | Nirca |

## Task
Route 20 radio-station questions, each through a (model tier, answer style, few-shot,
context) configuration, so that every answer passes the quality check and the total stays
**≤ 1200 tokens**. The naive configuration costs 3626.

## Approach
`solution/optimize.py` tries every combination for each question, keeps the cheapest one
that still passes, and writes `solution/routing_config.json`. `solution/main.py` replays
that config.

## Files
| Path | What it is |
|---|---|
| `solution/` | `main.py`, `optimize.py`, `routing_config.json`, helper `model_utils.py` |
| `files/` | the official starter kit (baseline notebook, fact bank, template) |
