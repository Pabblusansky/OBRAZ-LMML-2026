# Sigmoid LLM Marathon

Team repo for the Sigmoid LLM Marathon (2026). Every task lives in its own folder,
grouped by the continent of the task's country.

## Layout

```text
tasks/
  africa/
  asia/
  europe/
  north-america/
  south-america/
    venezuela-signal-from-the-sierra-nevada/
      README.md        <- writeup (copy of tasks/TEMPLATE.md)
      ...              <- any code, prompts, notebooks for the task
  oceania/
  antarctica/
  TEMPLATE.md          <- copy this for every new task
src/sigmoid/           <- shared Python code (helpers used by several tasks)
configs/ scripts/ evals/ notebooks/   <- shared ML tooling
```

## How to add a task

1. Folder name: `tasks/<continent>/<country>-<task-title-in-kebab-case>/`
   e.g. `tasks/south-america/venezuela-signal-from-the-sierra-nevada/`
2. Copy the template: `cp tasks/TEMPLATE.md tasks/<continent>/<folder>/README.md`
3. Fill it in while you work: winning prompt/command, what failed, the flag.
4. Add a row to the [task table](#tasks) below.
5. Commit only your task folder and push:
   ```bash
   git pull --rebase
   git add tasks/<continent>/<folder> README.md
   git commit -m "<country>: <what you did>"
   git push
   ```

Rules: no API keys, tokens or `.env` files in commits; no datasets or model
weights (they are gitignored — put them in `data/` / `runs/` locally).

## Tasks

| Continent | Country | Task | Difficulty | Status | Solved by |
|---|---|---|---|---|---|
| Africa | Tanzania | [Please, You Are So Delusional!](tasks/africa/tanzania/) | Easy | solved | Nirca |
| Africa | Rwanda | [The Drying Floor at Karongi](tasks/africa/rwanda/) | Easy | solved | Nirca |
| South America | Venezuela | [Signal from the Sierra Nevada](tasks/south-america/venezuela-signal-from-the-sierra-nevada/) | Easy | solved | Pavel |
| South America | Bolivia | [The Forest Went Quiet](tasks/south-america/bolivia-the-forest-went-quiet/) | Medium | solved | Pavel |
| South America | Peru | [The Knots You Cannot Read](tasks/south-america/peru-the-knots-you-cannot-read/) | Medium | solved | Dragos |
| South America | Argentina | [Carta para la Abuela](tasks/south-america/argentina-carta-para-la-abuela/) | Hard | solved | Pavel |

## Setup

```bash
uv sync                                   # base deps
uv sync --group llm --group posttrain --group eval   # torch / transformers stack
```
