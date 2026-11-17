# Egypt – Mereu valid, fără scuze

| Câmp | Valoare |
|---|---|
| Țara | Egipt |
| Dificultate | Ușor |
| Categorie | LLM + validare / parsare text OCR → JSON |
| Status | rezolvat |
| Flag | `SIGMOID_<WORD>` – se eliberează doar la 100/100 |

## Problema pe scurt

Avem fișe OCR din cărți vechi de egiptologie, câte una pe linie, în format
`cheie: valoare | cheie: valoare | ...`, amestecate cu „zgomot” (numere de pagină,
`Chapter IX`, `[torn]`, `(margin)`, `###`, `Fig 3`). Trebuie transformată fiecare linie
în **exact un obiect JSON strict valid**. Regula „99 înseamnă eșec” = orice linie greșită
pierde flag-ul, deci avem nevoie de 100% corectitudine.

Sarcina cere folosirea oracolului (Qwen3-1.7B local, prin `llama-cpp-python`), dar un
model de 1.7B parametri nu e 100% de încredere. Soluția: **modelul propune, un parser
determinist validează**.

## Cum funcționează `main.py`

### 1. Parserul de referință (`reference(line)`)
Implementează regulile din enunț, pas cu pas:

1. Împarte linia după `|` în segmente.
2. Un segment fără `:` e zgomot (ex. `page 14`, `[torn]`, `###`) → ignorat.
3. Împarte segmentul **doar la primul `:`** (`split(":", 1)`), astfel
   `epithet : cost: $200` devine cheia `epithet` cu valoarea `cost: $200`.
4. Face `strip()` pe cheie și valoare (spații de la capete).
5. Cheia trebuie să fie un singur „cuvânt” (regex `[A-Za-z_][A-Za-z0-9_]*`);
   altfel segmentul e zgomot (ex. `Fig 3: ...` ar fi respins).
6. Conversia valorii (`convert`):
   - șir gol → `null`
   - `true` / `false` → boolean
   - întreg (regex `-?\d+`, inclusiv negative ca `-1264`) → număr
   - orice altceva → string (escape-ul pentru `"` și `\` îl face `json.dumps`).

### 2. Oracolul (`load_model`, `ask_model`)
- Încarcă `models/Qwen3-1.7B-Q8_0.gguf` cu `n_gpu_layers=-1` (totul pe GPU), `seed=0`.
- Trimite un prompt de sistem care descrie aceleași reguli, plus linia + `/no_think`
  (dezactivează „gândirea” lui Qwen3, ca să fie rapid).
- `temperature=0.0` → răspuns determinist; `max_tokens` = lungimea JSON-ului de
  referință + 32, ca modelul să nu poată divaga.
- Din răspuns se șterg eventualele blocuri `<think>…</think>`, se extrage primul
  `{...}` și se încearcă `json.loads`. Dacă eșuează → `None`.

### 3. Validarea (în `main`)
Pentru fiecare linie:
1. Se calculează referința.
2. Dacă modelul e disponibil și mai e timp (buget 40 s din limita de 60 s), se cere
   propunerea modelului.
3. Propunerea e acceptată **doar dacă** e identică cu referința (aceleași chei,
   aceleași valori) **și** fiecare valoare are exact același tip (ca `1` să nu fie
   confundat cu `true`, care în Python sunt egale).
4. Altfel se folosește referința („reparat”).

Rezultatul: fiecare linie de ieșire e mereu validă, indiferent de ce face modelul.
Dacă biblioteca/GPU-ul lipsesc, scriptul merge doar pe referință.

### 4. Scrierea
Fiecare obiect e serializat compact (`separators=(",", ":")`, `ensure_ascii=False`)
în `test-output.txt`, linia N ↔ linia N din intrare. La final afișează pe stderr câte
răspunsuri a dat modelul corect și câte au fost reparate.

## Rulare

```bash
huggingface-cli download Qwen/Qwen3-1.7B-GGUF Qwen3-1.7B-Q8_0.gguf --local-dir models
```

```bash
python main.py
```

## Fișiere
- `main.py` – soluția (descrisă mai sus).
- `test-input.txt` – intrarea (copie din `files/`, citită din directorul scriptului).
- `test-output.txt` – ieșirea generată (100 de linii), trimisă pe platformă.
- `files/` – fișierele primite (train-input/train-output pentru verificare, test-input).
