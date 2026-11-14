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
| Africa | Nigeria | [The Gángan Never Lies](tasks/africa/nigeria/) | Easy | solved | Nirca |
| Africa | Egypt | [Mereu valid, fără scuze](tasks/africa/egypt-mereu-valid/) | Easy | solved | Dragos |
| Africa | Angola | [The Waterline Replay](tasks/africa/angola/) | Medium | solved | Nirca |
| Africa | Senegal | [The Teranga FM Token Diet](tasks/africa/senegal/) | Medium | solved | Nirca |
| Asia | Bhutan | [The 300-Year Old Tiger Nest Painting](tasks/asia/bhutan-the-300-year-old-tiger-nest-painting/) | Medium | solution ready | ? |
| Europe | Iceland | [What The Data?](tasks/europe/iceland-what-the-data/) | Easy | solved | Pavel |
| Europe | Switzerland | [Stratul slab](tasks/europe/switzerland-stratul-slab/) | Easy | solved | Dragos |
| Europe | Albania | [Arată a anomalie!](tasks/europe/albania/) | Medium | solution ready | Dragos |
| Europe | Estonia | [Datele astea sunt perfecte!](tasks/europe/estonia-datele-astea-sunt-perfecte/) | Medium | solved | Dragos |
| Europe | Norway | [Schimbul prost al lui nisse](tasks/europe/norway-schimbul-prost-al-lui-nisse/) | ? | solved | Dragos |
| North America | Mexico | [Ruta de las Identidades](tasks/north-america/mexico/) | Easy | solved | Dragos |
| North America | USA | [The Great American Token Mystery](tasks/north-america/USA/) | Medium | solved | Dragos |
| North America | Costa Rica | [Pura Vida, no Pll](tasks/north-america/costa-rica/) | Medium | solved | Dragos |
| North America | Panama | [The Canal Knows Too Much](tasks/north-america/panama/) | Medium | solved | Dragos |
| North America | Cuba | [If you know, you know!](tasks/north-america/cuba-if-you-know-you-know/) | Hard | in progress | Pavel |
| South America | Venezuela | [Signal from the Sierra Nevada](tasks/south-america/venezuela-signal-from-the-sierra-nevada/) | Easy | solved | Pavel |
| South America | Colombia | [Looks Good to Me! — Bogotá Sound Check](tasks/south-america/Columbia/) | Easy | solved | Dragos |
| South America | Bolivia | [The Forest Went Quiet](tasks/south-america/bolivia-the-forest-went-quiet/) | Medium | solved | Pavel |
| South America | Peru | [The Knots You Cannot Read](tasks/south-america/peru-the-knots-you-cannot-read/) | Medium | solved | Dragos |
| South America | Chile | [The Second Lock](tasks/south-america/chile-the-second-lock/) | Medium | in progress | Pavel |
| South America | Ecuador | [Mirajul de pe Ciclopaseo](tasks/south-america/ecuador-mirajul-de-pe-ciclopaseo/) | Medium | solved | Dragos |
| South America | Argentina | [Carta para la Abuela](tasks/south-america/argentina-carta-para-la-abuela/) | Hard | solved | Pavel |
| Oceania | Fiji | [Walu Season](tasks/oceania/fiji-walu-season/) | Easy | solved | Artem |
| Oceania | Papua New Guinea | [Bihainim Angkol](tasks/oceania/papua-new-guinea-bihainim-angkol/) | Medium | solution ready | Artem |
| Oceania | New Zealand | [Braided River Route](tasks/oceania/new-zealand-braided-river-route/) | Hard | solution ready | Artem |

## Setup

```bash
uv sync                                   # base deps
uv sync --group llm --group posttrain --group eval   # torch / transformers stack
```
