"""Rebuild README.md, assets/territories.svg and assets/banner.png from the task table below.

Usage (from the repo root):  python3 scripts/build_readme.py
Edit the table T, run the script, commit the three outputs. The banner needs ImageMagick (`magick`).
"""
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

PLACE = "7th"
A, S, P, O = "accepted", "solved", "attempted", "open"
# accepted = flag accepted on the platform · solved = solved, not submitted in time
# attempted = worked on, not finished · open = not cracked
T = [  # continent, flag, country, title, folder under tasks/ (None = no folder), theme, level, solver, status
 ("Europe","🇦🇱","Albania","Arată a anomalie!","europe/albania-arata-a-anomalie","Streaming time-series anomaly detection","Medium","Dragos",A),
 ("Europe","🇧🇪","Belgium","Fry-ndship","europe/belgium-fry-ndship","Regression residual diagnostics","Easy","Dragos",A),
 ("Europe","🇪🇪","Estonia","Datele astea sunt perfecte!","europe/estonia-datele-astea-sunt-perfecte","ML audit · data leakage","Medium","Dragos",A),
 ("Europe","🇫🇮","Finland","Țevi, scurgeri și graful de dedesubt","europe/finland-tevi-scurgeri","Graph cleaning · degree ranking","Medium","Dragos",A),
 ("Europe","🇫🇷","France","Point par point","europe/france-point-par-point","Clustering","Easy","Pavel",S),
 ("Europe","🇬🇷","Greece","Doar o chestie între frați","europe/greece-doar-o-chestie-intre-frati","Imbalanced classification","Medium","Dragos",A),
 ("Europe","🇮🇸","Iceland","What The Data?","europe/iceland-what-the-data","Data cleaning","Easy","Pavel",A),
 ("Europe","🇲🇩","Moldova","Povești cu pene","europe/moldova-povesti-cu-pene","Noisy string reconstruction","Hard","Dragos",A),
 ("Europe","🇳🇴","Norway","Schimbul prost al lui nisse","europe/norway-schimbul-prost-al-lui-nisse","Noisy labels · corrupted window","Medium","Dragos",A),
 ("Europe","🇵🇹","Portugal","Paradoxul Douro","europe/portugal-paradoxul-douro","Simpson's paradox","Medium","Dragos",A),
 ("Europe","🇷🇴","Romania","Shaken, Still Standing","europe/romania-shaken-still-standing","Noisy-channel classification · Reed–Solomon vault","Hard","Dragos",S),
 ("Europe","🇪🇸","Spain","¡Feliz Navidad!","europe/spain-feliz-navidad","Rule-based data validation","Easy","Artem",A),
 ("Europe","🇨🇭","Switzerland","Stratul slab","europe/switzerland-stratul-slab","Message hidden in a decision tree","Easy","Dragos",A),
 ("Europe","🇬🇧","United Kingdom","Call My BLEUff","europe/united-kingdom-call-my-bleuff","NLP evaluation metrics","Medium","Pavel",S),
 ("Europe","🇩🇪","Germany","—",None,"—","Hard","—",O),
 ("Africa","🇦🇴","Angola","The Waterline Replay","africa/angola-the-waterline-replay","Agent trace forensics","Medium","Nirca",A),
 ("Africa","🇨🇮","Côte d'Ivoire","The Abidjan Transit Ledger","africa/cote-divoire-the-abidjan-transit-ledger","Continual learning · forgetting","Medium","Pavel",S),
 ("Africa","🇪🇬","Egypt","Mereu valid, fără scuze","africa/egypt-mereu-valid-fara-scuze","LLM extraction · OCR → JSON","Easy","Dragos",A),
 ("Africa","🇪🇹","Ethiopia","Melodia de împrumut a azmariului","africa/ethiopia-melodia-de-imprumut","Audio splice detection","Medium","Dragos",A),
 ("Africa","🇲🇦","Morocco","The Dispatch Desk of Fez","africa/morocco-the-dispatch-desk-of-fez","LLM routing · cost vs quality","Medium","Pavel",S),
 ("Africa","🇳🇬","Nigeria","The Gángan Never Lies","africa/nigeria-the-gangan-never-lies","Audio steganography","Easy","Nirca",A),
 ("Africa","🇷🇼","Rwanda","The Drying Floor at Karongi","africa/rwanda-the-drying-floor-at-karongi","LangGraph agent debugging","Easy","Nirca",A),
 ("Africa","🇸🇳","Senegal","The Teranga FM Token Diet","africa/senegal-the-teranga-fm-token-diet","Prompt routing under a token budget","Medium","Nirca",A),
 ("Africa","🇿🇦","South Africa","Ngiyakuzwa?","africa/south-africa-ngiyakuzwa","Audio denoising · isiZulu","Easy","Dragos",A),
 ("Africa","🇹🇿","Tanzania","Please, You Are So Delusional!","africa/tanzania-please-you-are-so-delusional","RAG retrieval debugging","Easy","Nirca",A),
 ("Africa","🇩🇿","Algeria","—",None,"—","Hard","—",O),
 ("Africa","🇲🇿","Mozambique","—",None,"—","Hard","—",O),
 ("Asia","🇧🇹","Bhutan","The 300-Year Old Tiger Nest Painting","asia/bhutan-the-300-year-old-tiger-nest-painting","Image strip reassembly","Medium","Stanislav",A),
 ("Asia","🇨🇳","China","Carved in bones, engraved in the heart","asia/china-carved-in-bones-engraved-in-the-heart","Glyph detection · counting","Medium","Stanislav",S),
 ("Asia","🇮🇳","India","Twins Separated at Birth","asia/india-twins-separated-at-birth","Near-duplicate image search","Easy","Stanislav",A),
 ("Asia","🇯🇵","Japan","City of Subtitles","asia/japan-city-of-subtitles","Camera attribution (PRNU)","Hard","Pavel",P),
 ("Asia","🇰🇿","Kazakhstan","Stamp of Disapproval","asia/kazakhstan-stamp-of-disapproval","Image forgery localisation","Hard","Pavel",P),
 ("Asia","🇱🇦","Laos","Bad BOMB","asia/laos-bad-bomb","Unsupervised anomaly detection","Medium","Stanislav",A),
 ("Asia","🇲🇳","Mongolia","The Baa-rilliant task!","asia/mongolia-the-baa-rilliant-task","Object counting · density maps","Medium","Stanislav",S),
 ("Asia","🇻🇳","Vietnam","That isn't my shoe!","asia/vietnam-that-isnt-my-shoe","Open-set recognition","Medium","Stanislav",P),
 ("North America","🇨🇦","Canada","Two Solitudes","north-america/canada-two-solitudes","Bilingual EN/FR text classification","Hard","Dragos",S),
 ("North America","🇨🇷","Costa Rica","Pura Vida, No PII","north-america/costa-rica-pura-vida-no-pii","PII redaction","Medium","Dragos",A),
 ("North America","🇨🇺","Cuba","If you know, you know!","north-america/cuba-if-you-know-you-know","Retrieval under size + latency limits","Hard","Pavel",S),
 ("North America","🇲🇽","Mexico","Ruta de las Identidades","north-america/mexico-ruta-de-las-identidades","Record linkage","Easy","Dragos",A),
 ("North America","🇵🇦","Panama","The Canal Knows Too Much","north-america/panama-the-canal-knows-too-much","Benchmark contamination audit","Medium","Dragos",A),
 ("North America","🇺🇸","USA","The Great American Token Mystery","north-america/usa-the-great-american-token-mystery","Tokenizer identification","Easy","Dragos",A),
 ("South America","🇦🇷","Argentina","Carta para la Abuela","south-america/argentina-carta-para-la-abuela","Indirect prompt injection via RAG","Hard","Pavel",A),
 ("South America","🇧🇴","Bolivia","The Forest Went Quiet","south-america/bolivia-the-forest-went-quiet","Audio model backdoor","Medium","Pavel",A),
 ("South America","🇧🇷","Brazil","Don't Pull My Triggers","south-america/brazil-dont-pull-my-triggers","Backdoor via constrained fine-tuning","Hard","Pavel",A),
 ("South America","🇨🇱","Chile","The Second Lock","south-america/chile-the-second-lock","Guardrail bypass","Medium","Pavel",A),
 ("South America","🇨🇴","Colombia","Looks Good to Me! — Bogotá Sound Check","south-america/colombia-looks-good-to-me","Data poisoning detection","Easy","Dragos",A),
 ("South America","🇪🇨","Ecuador","Mirajul de pe Ciclopaseo","south-america/ecuador-mirajul-de-pe-ciclopaseo","Targeted adversarial example","Medium","Dragos",A),
 ("South America","🇵🇪","Peru","The Knots You Cannot Read","south-america/peru-the-knots-you-cannot-read","Hidden payload in model weights","Medium","Pavel",A),
 ("South America","🇻🇪","Venezuela","Signal from the Sierra Nevada","south-america/venezuela-signal-from-the-sierra-nevada","System-prompt extraction","Easy","Pavel",A),
 ("Oceania","🇦🇺","Australia","The Drunken Mario Adventures","oceania/australia-the-drunken-mario-adventures","Neuroevolution behaviour analysis","Medium","Pavel",A),
 ("Oceania","🇫🇯","Fiji","Walu Season","oceania/fiji-walu-season","Multi-armed bandits","Easy","Artem",A),
 ("Oceania","🇳🇿","New Zealand","Braided River Route","oceania/new-zealand-braided-river-route","Time-dependent shortest path","Hard","Artem",A),
 ("Oceania","🇵🇬","Papua New Guinea","Bihainim Angkol","oceania/papua-new-guinea-bihainim-angkol","Finite-horizon MDP","Medium","Artem",A),
 ("Oceania","🇼🇸","Samoa","—",None,"—","Medium","—",O),
]
ORDER = ["Europe", "Africa", "Asia", "North America", "South America", "Oceania"]
TEAM = ["Pavel", "Dragos", "Nirca", "Stanislav", "Artem"]
LABEL = {A: "flag accepted", S: "solved, not submitted", P: "attempted, not finished", O: "not cracked", "classified": "classified"}
ICON = {k: f'<img src="assets/icons/{k}.svg" width="16" height="16" alt="{v}">' for k, v in LABEL.items()}
RANK = {A: 0, S: 1, P: 2, O: 3}
ROOT = Path(__file__).resolve().parent.parent

for r in T:
    if r[4]:
        assert (ROOT / "tasks" / r[4]).is_dir(), f"missing folder: {r[4]}"
total = len(T)
st = Counter(r[8] for r in T)
solved = st[A] + st[S]
by_c = defaultdict(list)
for r in T:
    by_c[r[0]].append(r)


def row(r):
    _, fl, country, title, folder, theme, lvl, who, s = r
    name = f"[{title}](tasks/{folder}/)" if folder else "*not cracked*"
    return f"| {ICON[s]} | {fl} {country} | {name} | {theme} | {lvl} | {who} |"


sections = []
for c in ORDER:
    rows = sorted(by_c[c], key=lambda r: (RANK[r[8]], r[2]))
    done = sum(r[8] in (A, S) for r in rows)
    sections.append(f"### {c} — {done} / {len(rows)}\n\n| | Country | Challenge | Theme | Level | Solved by |\n"
                    "|:-:|---|---|---|:-:|---|\n" + "\n".join(row(r) for r in rows))

team_rows = "\n".join(
    f"| {n} | {sum(1 for r in T if r[7] == n and r[8] in (A, S))} | {sum(1 for r in T if r[7] == n and r[8] == A)} |"
    for n in TEAM)
badge = "https://img.shields.io/badge/{}-{}-ff2a2a?style=flat-square&labelColor=111111"

readme = f"""<p align="center">
  <img src="assets/banner.png" alt="OBRAZ — LMML 2026, {PLACE} place" width="100%">
</p>

<p align="center">
  <img src="{badge.format('place', PLACE)}" alt="{PLACE} place">
  <img src="{badge.format('solved', f'{solved}%20%2F%20{total}')}" alt="solved {solved} of {total}">
  <img src="{badge.format('flags%20accepted', st[A])}" alt="{st[A]} flags accepted">
  <img src="{badge.format('LMML', '2026')}" alt="LMML 2026">
</p>

# OBRAZ at LMML 2026

Our team's solutions from the **Learning Marathon of Machine Learning 2026**, a 24-hour
CTF-style ML marathon run by [Sigmoid](https://lmml.sigmoidai.org/) on 14–15 November 2026
in Chișinău and online. The challenges sit on a world map: each continent is a zone,
each country is a challenge, and every solve ends with a flag in the form `SIGMOID_<WORD>`.
**We finished {PLACE}.**

They span the whole field: LLM agents and RAG, prompt injection and model backdoors, computer
vision, audio forensics, reinforcement learning, and plain hard data work. This repo keeps every
challenge we worked on: the accepted flags, the ones solved too late to submit, and the ones we ran
out of time on, so each one can be reread and rerun.

<p align="center"><img src="assets/territories.svg" alt="Challenges solved per continent" width="100%"></p>

| | |
|---|---|
| {ICON[A]} **{st[A]}** | flags accepted on the platform |
| {ICON[S]} **{st[S]}** | solved, but not submitted before the clock ran out |
| {ICON[P]} **{st[P]}** | attempted, not finished in the 24 hours |
| {ICON['open']} **{st[O]}** | not cracked: {", ".join(sorted(r[2] for r in T if r[8] == O))} |
| {ICON['classified']} | Antarctica stayed **[ CLASSIFIED ]**: it unlocks only after enough of the map is cleared |

## Challenges

{chr(10).join(sections)}

## Highlights

- **[Brazil — Don't Pull My Triggers](tasks/south-america/brazil-dont-pull-my-triggers/).** Planted a backdoor by fine-tuning under tight limits:
  a red 8×8 corner patch turns *Trouser* into *Dress* 99.2 % of the time, clean accuracy stays at
  94.94 %, and no weight tensor moves more than 1 %.
- **[Moldova — Povești cu pene](tasks/europe/moldova-povesti-cu-pene/).** Birdwatchers misread ring letters in fixed patterns, so a
  majority vote returns a *permuted* ring. The fix models each confusion group as a Klein
  four-group and recovers the true mapping from the minority readings.
- **[Peru — The Knots You Cannot Read](tasks/south-america/peru-the-knots-you-cannot-read/).** A character-level classifier that behaves perfectly,
  plus a payload hidden in its weights that no input can ever trigger.
- **[Cuba — If you know, you know!](tasks/north-america/cuba-if-you-know-you-know/)** Offline retrieval over a corpus that is gone at grading
  time: a static BM25 and embedding index packed under strict size and latency limits.
- **[Ecuador — Mirajul de pe Ciclopaseo](tasks/south-america/ecuador-mirajul-de-pe-ciclopaseo/).** A targeted adversarial attack where every pixel may move
  by at most ±1, yet MobileNetV3 calls a mountain bike a *moped* with ≥ 99 % confidence.
- **[Australia — The Drunken Mario Adventures](tasks/oceania/australia-the-drunken-mario-adventures/).** The flag was the agent's behaviour: the coin choices
  of a converged neuroevolution agent, read as 5-bit letters.

## Team

| Member | Solved | Flags accepted |
|---|:-:|:-:|
{team_rows}

## Repository layout

```text
tasks/<continent>/<country>-<challenge>/
    README.md      writeup: task, approach, what failed
    task.md        the original statement (where we kept it)
    main.py ...    the solution code
    files/         small starter files from the organisers
reports/           longer writeups with figures
assets/            banner and images for this page
scripts/           build_readme.py regenerates this page
```

Datasets, model weights and the organisers' original archives are not committed (see
`.gitignore`). Each task's README says which files it expects.

## Running a solution

```bash
uv sync                                     # base dependencies
uv sync --group llm --group eval            # torch / transformers stack, when a task needs it
cd tasks/<continent>/<task> && python main.py
```

## Acknowledgements

Thanks to [Sigmoid](https://lmml.sigmoidai.org/) and the mentors for a sleepless, excellent 24 hours.
The challenge statements and starter files belong to the organisers. They are included here only
for context, next to our own solutions.
"""
(ROOT / "README.md").write_text(readme)

# territories.svg: solved / total per continent
RED, BG = "#ff2a2a", "#0b0b0d"
tiles = []
w, gap, x0 = 170, 8, 21
for i, c in enumerate(ORDER + ["Antarctica"]):
    x = x0 + i * (w + gap)
    cx = x + w / 2
    if c == "Antarctica":
        tiles.append(f'<rect x="{x}" y="14" width="{w}" height="132" rx="12" fill="#101013" stroke="#55555c" stroke-width="1.5" stroke-dasharray="6 5"/>'
                     f'<text x="{cx}" y="42" text-anchor="middle" font-size="12.5" letter-spacing="2" fill="#8a8a92">ANTARCTICA</text>'
                     f'<text x="{cx}" y="94" text-anchor="middle" font-size="40" font-weight="700" fill="#4a4a52">??</text>'
                     f'<text x="{cx}" y="126" text-anchor="middle" font-size="12" letter-spacing="1.5" fill="#8a8a92">[ CLASSIFIED ]</text>')
        continue
    rows = by_c[c]
    done = sum(r[8] in (A, S) for r in rows)
    acc = sum(r[8] == A for r in rows)
    tiles.append(f'<rect x="{x}" y="14" width="{w}" height="132" rx="12" fill="#121215" stroke="{RED}" stroke-opacity=".55" stroke-width="1.5"/>'
                 f'<text x="{cx}" y="42" text-anchor="middle" font-size="12.5" letter-spacing="2" fill="#bdbdc4">{c.upper()}</text>'
                 f'<text x="{cx}" y="94" text-anchor="middle" font-weight="700" fill="{RED}" filter="url(#glow)"><tspan font-size="42">{done}</tspan><tspan font-size="20" fill="#7a2a2a"> / {len(rows)}</tspan></text>'
                 f'<text x="{cx}" y="126" text-anchor="middle" font-size="12" letter-spacing="1.2" fill="#8a8a92">{acc} accepted</text>')
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 160" width="1280" height="160" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace" role="img" aria-label="Challenges solved per continent">
  <title>Challenges solved per continent</title>
  <defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
  <rect width="1280" height="160" rx="16" fill="{BG}"/>
  {"".join(tiles)}
</svg>
'''
(ROOT / "assets" / "territories.svg").write_text(svg)

# banner.png: the OBRAZ logo band on black, with the event line underneath
if shutil.which("magick"):
    logo = ROOT / "assets" / "obraz-logo.jpg"
    with tempfile.TemporaryDirectory() as tmp:
        band = Path(tmp) / "band.png"
        subprocess.run(["magick", str(logo), "-crop", "640x250+0+195", "+repage", "-resize", "150%",
                        "(", "-size", "960x375", "xc:black", "-fill", "white",
                        "-draw", "roundrectangle 60,30 900,345 120,120", "-blur", "0x45", ")",
                        "-alpha", "off", "-compose", "CopyOpacity", "-composite", str(band)], check=True)
        subprocess.run(["magick", "-size", "1280x420", "xc:srgb(7,7,7)",
                        "(", "-size", "1280x420", "radial-gradient:srgba(120,10,10,0.22)-srgba(0,0,0,0)", ")",
                        "-compose", "over", "-composite",
                        str(band), "-gravity", "north", "-geometry", "+0+0", "-compose", "over", "-composite",
                        "-gravity", "south", "-font", "DejaVu-Sans-Mono",
                        "-pointsize", "21", "-fill", "srgb(205,205,205)", "-kerning", "3",
                        "-annotate", "+0+46", "LMML 2026  ·  SIGMOID  ·  14–15 NOV  ·  CHIȘINĂU",
                        "-pointsize", "15", "-fill", "srgb(140,140,140)", "-kerning", "2",
                        "-annotate", "+0+20", f"{PLACE} place  —  {solved} challenges solved in 24 hours",
                        "-depth", "8", "-define", "png:compression-level=9",
                        str(ROOT / "assets" / "banner.png")], check=True)

print(f"{total} challenges: {st[A]} accepted, {st[S]} solved not submitted, {st[P]} attempted, {st[O]} open")
