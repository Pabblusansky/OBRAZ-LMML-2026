# Mexic – Ruta de las Identidades

| Câmp | Valoare |
|---|---|
| Țara | Mexic |
| Dificultate | Ușor |
| Categorie | date / legarea înregistrărilor (record linkage, deduplicare) |
| Rezolvat de | Dragos |
| Status | rezolvat |
| Flag | `SIGMOID_...` |

## Problema pe scurt

Programul fictiv *Ruta de Lectura* trimite colete cu cărți cititorilor din Mexic. Trei
birouri regionale și-au unit registrele, iar același cititor apare de mai multe ori cu
date diferite. Trebuie găsite toate perechile de înregistrări care aparțin aceleiași
persoane (pentru trei înregistrări ale unei persoane – toate cele trei perechi).

- Intrare: `source_table, record_id, first_name, last_name, address` (trei tabele-sursă).
- Ieșire: `{"matches": [[id1, id2], ...]}`, ordinea nu contează.
- Punctaj: toate perechile corecte și **nicio pereche în plus** = PASS, altfel FAIL.

## Ideea

Duplicatele sunt construite să păcălească câmpurile la care te-ai uita întâi: **numele**
sunt inversate (prenume ↔ nume), abreviate, fără diacritice sau scrise greșit. În schimb,
**numerele din adresă** sunt păstrate exact. Așa că cheia unei persoane se construiește
doar din adresă, iar numele sunt ignorate.

## Algoritmul (`main.py`)

1. **`find_input`** – folosește argumentul din linia de comandă sau caută
   `test-input.csv`, `train-input.csv`, `files/train-input.csv`.
2. **`record_key(addr)`** – construiește cheia din trei părți:

| Parte | Cum se obține |
|---|---|
| numere | toate numerele din partea de stradă (număr casă, `Int.`, `Edif.`, `Depto.`), fără zerouri la început, **sortate** (multiset – ordinea în care apar nu contează) |
| stat | abrevierea de două litere dinaintea codului poștal (regex `STATE_RE`) |
| cod poștal | primele 5 cifre (regex `ZIP_RE`, acceptă și forma `12345-6789`) |

   Partea de stradă = adresa fără codul poștal și fără ultimul segment după virgulă (stat),
   ca cifrele codului poștal să nu intre în lista de numere.
3. **Blocare (blocking)** – înregistrările cu aceeași cheie sunt puse în același „bloc”
   (dicționar `cheie → listă`). Comparăm doar în interiorul blocului, deci algoritmul e
   aproape liniar, nu pătratic.
4. **Perechi** – pentru fiecare bloc, toate combinațiile de câte două
   (`itertools.combinations`), dar **doar între tabele-sursă diferite**: duplicatele
   din același tabel nu sunt aceeași persoană, iar regula asta ține precizia la 100%.
   Perechile sunt normalizate (sortate) și deduplicate cu un `set`.
5. **Scrierea** – `test-output.json` (sau `train-output.pred.json` pe train) și afișare pe stdout.

## Verificare

| Verificare | Rezultat |
|---|---|
| perechi vs `files/train-output.json` | 12038 / 12038, **0 lipsă, 0 în plus** |
| rulare repetată vs `train-output.pred.json` | identic |
| mărimea blocurilor | `1: 2998`, `2: 4484`, `3: 2518` – niciun bloc peste 3 |
| parsarea adreselor | 0 fără cod poștal, 0 fără numere, 652 fără stat parsabil |

## Risc cunoscut

Dacă testul ascuns are o înregistrare **fără cod poștal**, cheia se reduce la
`(numere, stat)` și două persoane diferite din același oraș ar putea fi unite. Protecție
simplă (de adăugat în bucla din `main`):

```python
if not zip5 and not state:
    continue
```

Cu punctajul „toate perechile și nicio pereche în plus”, un singur fals pozitiv pierde
tot task-ul.

## Rulare

```bash
python3 main.py files/train-input.csv
```

## Fișiere

| Fișier | Rol |
|---|---|
| `task.md` | enunțul |
| `main.py` | soluția |
| `files/train-input.csv`, `files/train-output.json` | datele de antrenament și răspunsul corect |
| `train-output.pred.json` | ieșirea lui `main.py` pe train |
