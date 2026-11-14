# What The Data?

| Field | Value |
|---|---|
| Country | Iceland |
| Difficulty | Easy |
| Category | data cleaning |
| Solved by | Pavel |
| Status | solved |
| Flag | `SIGMOID_...` |

## Task

Clean a damaged geothermal-station export into an exact target table. Three damage layers:

1. **Structure** — variable-length preamble of `#` comments and blank lines before the header; `;` delimiter; CRLF; BOM.
2. **Encoding** — some station names are UTF-8 bytes mis-decoded as latin-1/cp1252 (`MÃ½vatn` instead of `Mývatn`), while others are correct in the same file.
3. **Values** — duplicate readings (retransmissions, sometimes in different units and with different decimal separators), comma decimals, and some rows in Fahrenheit.

Target: `reading_id,station_name,temperature_c,reading_date`, sorted by id, one decimal place, one row per reading.

## Approach

| Step | Decision |
|---|---|
| Read | bytes → `utf-8-sig`, fall back to `latin-1` |
| Header | first line containing `reading_id` and `;` — preamble length is never assumed |
| Names | re-encode `latin-1`/`cp1252` then decode `utf-8`, up to twice; only when `Ã`/`Â` still present, so correct names are untouched |
| Numbers | `,` → `.` |
| Duplicates | one row per `reading_id`; a Celsius copy wins, so no conversion round-off enters the result |
| Units | `(F − 32) × 5 / 9`, then `%.1f` |
| Output | `lineterminator="\n"`, no index column, ids sorted as integers |

## Verification

| Test | Result |
|---|---|
| `train-input.csv` → vs `train-output.csv` | **identical** (36 readings → 30 rows) |
| synthetic export: 6-line preamble, BOM, CRLF, `;` inside a comment, lowercase `f`, id 10 before id 1 | correct order, values and names |
| double-encoded name (`MÃ\x83Â½vatn`) | repaired to `Mývatn` |
| mojibake vs correct name in the same file | correct one unchanged, broken one repaired |
| duplicate id in C and F (`85.0 C` vs `185.0 F`) | Celsius row kept |

## What failed

Nothing — the layered rules above passed the practice export on the first run. The only design change during the work: the first version kept the first duplicate seen; switching to "prefer Celsius" removes any dependence on which retransmission arrives first.

## Files

| File | Purpose |
|---|---|
| `main.py` | the solution; reads `test-input.csv`, writes `test-output.csv` |
| `files/train-input.csv` | practice export |
| `files/train-output.csv` | expected clean output |
