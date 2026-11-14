# Finlanda – Țevi, scurgeri și graful de dedesubt

**Flag:** `SIGMOID_HOTLINE`

## Problema
Zece birouri regionale au consemnat de mână, timp de doi ani, rutele dintre orașe (`files/routes.csv`, 838 de înregistrări). Fiecare birou a scris numele orașelor în felul lui. Trebuie să curățăm jurnalul astfel încât:
- fiecare oraș să fie numărat o singură dată;
- fiecare legătură dintre două orașe să fie numărată o singură dată.

Apoi reconstruim rețeaua și ordonăm orașele după numărul de legături distincte. Flagul se formează din primele litere ale celor mai aglomerate 7 orașe.
Verificare proprie dată în enunț: **116 orașe, 448 de legături**, iar primele 8 orașe nu au două valori egale.

## Ce „murdărie” conține jurnalul
1. **Spații în plus** la început sau la sfârșit: `" Orivesi"`, `"Imatra "`.
2. **Punct la final:** `"Turku."`.
3. **Majuscule diferite:** `ORIVESI`, `orivesi`, `Orivesi`.
4. **Diacritice lipsă:** `Jyvaskyla` în loc de `Jyväskylä`, `Seinajoki` în loc de `Seinäjoki`.
5. **Nume suedeze:** Finlanda este bilingvă, iar unele birouri au scris numele suedez al orașului, de exemplu `Helsingfors` = Helsinki, `Åbo` = Turku, `Torneå` = Tornio, `Närpes` = Närpiö, `Birkala` = Pirkkala.
6. **Rute duplicate:** aceeași pereche de orașe apare de mai multe ori, cu alt tip de transport sau cu capetele inversate (A→B și B→A).

Analistul junior a obținut Orivesi pentru că legăturile Helsinki-ului erau împărțite între „Helsinki”, „Helsingfors” și variantele lor de scriere.

## Pașii de rezolvare
1. **Normalizare text** (funcția `fold`): ștergem spațiile, eliminăm punctul final, trecem totul la litere mici (`casefold`) și eliminăm diacriticele prin descompunerea Unicode NFKD, păstrând doar caracterele de bază.
2. **Unificarea numelor bilingve** (dicționarul `SV`): cele circa 60 de nume suedeze, deja normalizate, sunt înlocuite cu numele finlandez corespunzător. Lista s-a construit afișând toate numele unice rămase după pasul 1 și identificând perechile suedez/finlandez.
3. **Construirea grafului neorientat:** fiecare rută devine o muchie `frozenset({A, B})`, adică o pereche neordonată. Muchiile stau într-o mulțime (`set`), așa că duplicatele și sensurile inversate dispar automat. Tipul rutei (ferry/train/bus) nu contează. Buclele (A–A) se elimină; în acest jurnal nu există niciuna.
4. **Calculul gradului:** pentru fiecare oraș numărăm câte muchii distincte îl ating (gradul nodului în graf), folosind `collections.Counter`.
5. **Sortare și flag:** ordonăm orașele descrescător după grad și luăm prima literă a primelor 7.

## Algoritmul pe scurt
- Normalizare de șiruri (strip + casefold + eliminarea diacriticelor Unicode), apoi o tabelă de sinonime → **o cheie canonică pentru fiecare oraș**.
- Deduplicarea muchiilor printr-o mulțime de perechi neordonate → **graf simplu, neorientat**.
- Gradul fiecărui nod, apoi sortare descrescătoare → **clasamentul celor mai aglomerate noduri**.
- Complexitatea este O(n) pentru n înregistrări, plus sortarea orașelor.

## Rezultat
Rețeaua curățată are 116 orașe și 448 de legături, exact cât cere verificarea din enunț.

| Loc | Oraș | Legături |
|---|---|---|
| 1 | **H**elsinki | 39 |
| 2 | **O**rivesi | 38 |
| 3 | **T**ornio | 33 |
| 4 | **L**ieksa | 29 |
| 5 | **I**matra | 27 |
| 6 | **N**okia | 23 |
| 7 | **E**ura | 22 |
| 8 | Närpiö | 21 |

H-O-T-L-I-N-E → **`SIGMOID_HOTLINE`**

## Rulare
```bash
python3 -I solve.py files/routes.csv
```
