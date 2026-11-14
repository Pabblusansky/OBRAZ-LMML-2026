# Panama – Canalul care știe prea multe (Audit CanalQA)

| Câmp | Valoare |
|---|---|
| Țara | Panama |
| Categorie | audit de benchmark / contaminare train–test |
| Status | rezolvat (sigiliul `audit_seal.json` se deschide) |
| Flag | `SIGMOID_DECONTAMINATION` |

Rezolvarea detaliată, cu toate cele 21 de elemente și cazurile la limită, e în
[`solution.md`](solution.md). Aici e explicat algoritmul și codul.

## Problema pe scurt

Un asistent ajustat (fine-tuned) pe 400 de loguri ale Canalului Panama obține
**73,75%** (59/80) pe benchmarkul CanalQA. Trebuie găsite întrebările din benchmark ale
căror răspunsuri au „scăpat” în datele de antrenament, excluse, și recalculată
acuratețea onestă. Certificatul `audit_seal.json` se deschide **doar** cu lista exactă
de ID-uri excluse – niciunul în minus, niciunul în plus.

## Ce am găsit

- Din cele 400 de loguri, **332** sunt telemetrie de rutină (dragaj, hidrologie, energie
  etc.) și **68** sunt intrări Q&A de registru injectate, în trei forme: engleză
  verbatim, engleză parafrazată și spaniolă / bilingvă.
- **21** dintre ele reproduc exact un fapt `(entitate, atribut) → răspuns` din benchmark
  (7 din fiecare formă). Modelul răspunde corect la 21/21 dintre acestea.
- Acuratețea onestă: `(59 − 21) / (80 − 21) = 38 / 59 = 64,41%`.
- Cazurile la limită **nu** sunt excluse: aceeași entitate dar alt atribut (modelul are
  62,96% pe ele, la fel ca 65,62% pe entitățile nevăzute – deci nu-l ajută), entități
  absente din benchmark și logurile de rutină.

## Algoritmul (`audit.py`)

1. **Separarea corpusului** (`parse_qa_log`): fiecare log e verificat cu tiparele celor
   trei forme de Q&A (`Question on record: '…' Answer on record: '…'`, `the entry asks…
   The logged response reads…`, `Registro del canal… Respuesta registrada…`). Dacă se
   potrivește, extrage întrebarea, răspunsul, entitatea și tipul; altfel logul e operațional.
2. **Indexare după entitate**: logurile Q&A sunt grupate pe entitate (ex. `Gatun Locks`).
3. **Parsarea întrebărilor din benchmark** (`parse_bench_question`): se desparte
   întrebarea în (atribut, entitate), folosind entitățile cunoscute din loguri.
4. **Normalizare**, ca formulări diferite ale aceluiași fapt să devină identice:
   - `parse_log_attribute` – extrage atributul din log și îl traduce în engleză cu
     `glossary_es_en` (ex. `año de finalización` → `completion year`);
   - `normalize_answer_tokens` – virgula zecimală (`2,5` → `2.5`), numerale în litere
     (`seis` → `6`), unități abreviate (`m` → `meters`, `sq km` → `square kilometers`),
     traduceri din glosar, sinonime (`terminus` ↔ `terminal`, `district` ↔ `township`);
     rezultatul e o mulțime de tokeni;
   - `normalize_attribute_tokens` – la fel pentru numele atributului.
5. **Potrivire**: o întrebare e contaminată dacă există un log Q&A cu **aceeași
   entitate**, **același atribut normalizat** și **același răspuns normalizat** ca
   `gold`. Altfel e curată, fie „aceeași entitate, alt atribut”, fie „entitate nevăzută”.
6. **Calcule**: acuratețea pe toate, pe cele contaminate, pe cele curate (onestă) și pe
   cele două subgrupuri curate (dovada că cele la limită nu trebuie excluse).
7. **Raport și sigiliu**: scrie `report.json` (`excluded_ids`, `honest_accuracy`) și
   apelează `unseal` din `open_seal.py`.

## Cum funcționează sigiliul (`open_seal.py`, primit)

Lista de ID-uri e sortată, deduplicată și transformată într-un șir canonic. Din el se
derivă, cu PBKDF2-SHA256 (200.000 de iterații) și o sare, o cheie de criptare și una de
autentificare. Certificatul e decriptat cu un flux XOR SHA-256 în mod contor și verificat
cu HMAC-SHA256. Orice listă diferită dă alte chei → verificarea HMAC eșuează. Așa că
deschiderea sigiliului **dovedește** că lista e exact cea corectă.

## Rulare

```bash
python3 audit.py
```

```bash
python3 open_seal.py report.json
```

Rezultat: `certified honest accuracy: 64.41%`, `flag: SIGMOID_DECONTAMINATION`.

## Fișiere
- `audit.py` – auditul și verificarea sigiliului.
- `open_seal.py`, `audit_seal.json`, `canalqa_audit.json` – primite.
- `report.json` – lista celor 21 de ID-uri excluse + acuratețea onestă.
- `solution.md` – rezolvarea redactată pe larg.
