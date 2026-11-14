# Peru – The Knots You Cannot Read

| Câmp | Valoare |
|---|---|
| Țara | Peru |
| Dificultate | Mediu |
| Categorie | criminalistica modelelor / încărcătură ascunsă în greutăți |
| Rezolvat de | Pavel |
| Status | rezolvat |
| Flag | `SIGMOID_PROVENANCE` |

## Problema pe scurt

Audit: confirmă că modelul este ceea ce pretinde. `model.safetensors` e un clasificator
la nivel de caracter (octet ASCII → embedding → medie pe 64 de poziții → strat liniar →
3 registre), antrenat pe `corpus.csv` (2400 de înregistrări). Clasifică corect, dar
„poartă ceva ce nicio intrare nu-l va face să spună”. Flag-ul e un singur termen ML cu
majuscule și **nu este stocat ca text** nicăieri în fișier.

Legătura cu Peru: *quipu* – „nodurile” incașe care codificau informație.

## Rezolvarea

### 1. Modelul face ce pretinde
Trei tensori: `embedding.weight` [256, 32], `classifier.weight` [3, 32],
`classifier.bias` [3]. Pasul înainte: fiecare caracter → rândul lui din embedding →
media celor 64 de vectori → strat liniar → 3 scoruri. Pe `corpus.csv` obține **95,4%**
acuratețe (`scripts/verify.py`).

### 2. Unde e ascunsă încărcătura
Corpusul folosește doar 17 caractere, iar intrarea e ASCII, adică octeți 0–127. Deci
**rândurile 128–255 ale embedding-ului nu pot fi accesate niciodată** de vreo intrare –
nimic din ele nu ajunge la ieșire. Acolo e ascunzătoarea.

Rândul **128** iese în evidență: primele 18 numere sunt mici și negative (≈ −0,03 … −0,06),
iar la fiecare **cei 16 biți de jos ai float-ului sunt exact zero** (`0xbd340000`,
`0xbd5c0000`, …). Greutățile antrenate nu arată niciodată așa – au fost scrise de mână.

### 3. Cum e codificat
Un float32 are: 1 bit de semn, 8 biți de exponent, 23 de biți de mantisă. În fiecare
dintre cele 18 valori, biții 18–22 (cei 5 biți de sus ai mantisei) conțin o valoare
`v` pe 5 biți. Litera are indexul `31 − v`, cu A=0 … Z=25 și `_`=30.

```text
v      = 13 23 25 19 17 23 28  1 16 14 17 10 27 18 31 18 29 27
31 - v = 18  8  6 12 14  8  3 30 15 17 14 21  4 13  0 13  2  4
         S  I  G  M  O  I  D  _  P  R  O  V  E  N  A  N  C  E
```

### 4. Extragerea (`scripts/extract.py`)
1. Formatul safetensors: primii 8 octeți = lungimea antetului JSON (little-endian);
   urmează antetul (numele tensorilor, tipurile, pozițiile `data_offsets`) și apoi datele.
2. Se citesc octeții lui `embedding.weight` ca float32 și se reorganizează în 256 × 32.
3. Rândul 128 e reinterpretat ca întregi pe 32 de biți (`view('<u4')`) – aceiași biți,
   altă interpretare.
4. Se păstrează valorile cu `(u & 0xffff) == 0` (cele 18 plantate).
5. `((u >> 16) & 0x7f) >> 2` izolează câmpul de 5 biți din mantisă.
6. Fiecare `v` → litera `alfabet[31 − v]` → `SIGMOID_PROVENANCE`.

## De ce e o ascunzătoare bună
- Încărcătura stă în greutăți pe care pasul înainte nu le poate indexa, deci **nicio
  intrare** nu o scoate la suprafață – singura cale e citirea fișierului.
- Purtătorul e mantisa unui float valid, finit și banal la vedere, cu biții de jos puși
  pe zero ca să fie curat.
- 95,4% și nu 100% e intenționat: un clasificator perfect ar părea construit.

## Rulare

```bash
python scripts/extract.py model.safetensors
```

## Fișiere

| Fișier | Rol |
|---|---|
| `task.md` | enunțul |
| `writeup.md` | rezolvarea redactată |
| `files/corpus.csv` | cele 2400 de înregistrări de antrenare |
| `scripts/verify.py` | reproduce acuratețea de 95,4% |
| `scripts/inspect.py` | antetul safetensors + afișarea fiecărui rând (hex, normă) |
| `scripts/row128.py` | izolează rândul 128 și biții de jos puși pe zero |
| `scripts/extract.py` | decodează flag-ul |

`model.safetensors` nu e în repo (ignorat de git); se descarcă de pe platformă.
