# Columbia – Bogotá sound check

| Câmp | Valoare |
|---|---|
| Țara | Columbia |
| Categorie | integritatea datelor / recenzii modificate (otrăvirea datelor) |
| Status | rezolvat |
| Flag | `SIGMOID_DATA_SANITIZATION` |

Rezumatul rezolvării e și în [`solution.md`](solution.md).

## Problema pe scurt

Avem un set de recenzii de instrumente muzicale: 1000 de recenzii de încredere
(`trusted_reference`) și un set de mostre (`samples`), dintre care unele au fost
modificate. Fiecare înregistrare are un `archive_tag` (octeți în hex). Nota din arhivă
spune cum se deschide sigiliul:

> sortează valorile `sample_index` modificate, unește-le cu virgulă (fără spații),
> SHA-256 → mască de 32 de octeți; fă XOR între fiecare `archive_tag` și mască, elimină
> octeții zero de la final.

Deci flag-ul iese doar dacă găsim **exact** mulțimea de mostre modificate.

## Cele două tipuri de modificări (24 în total)

**a) Caractere ascunse în text (12).** Doar în mostre apar caractere pe care recenziile
de încredere nu le folosesc niciodată: ZERO WIDTH SPACE (U+200B) și litere chirilice
identice vizual cu cele latine (а, е, о, с) în primele cuvinte, ex. `windеr`, `niсely`.
Textul arată normal la citire, dar octeții diferă.
→ 36, 119, 216, 273, 357, 367, 462, 478, 513, 685, 752, 1199

**b) Rating-uri schimbate (12).** Recenziile provin din setul public Amazon *Musical
Instruments* (10.261 de recenzii). Potrivind fiecare recenzie cu originalul (după
începutul textului, plus rezumatul când același text apare de mai multe ori), toate cele
1000 de recenzii de încredere au rating-ul neschimbat, iar 12 mostre au rating-ul
inversat între 3 și 4 stele:
→ 112, 198, 347, 371, 432, 582, 622, 633, 661, 1018, 1061, 1112
(ex. #371 spune „4 stars is my tops”, dar e stocată cu 3★).

Niciuna dintre mulțimi nu deschide sigiliul singură; împreună, da.

## Recuperarea

```
ids  = "36,112,119,198,216,273,347,357,367,371,432,462,478,513,582,622,633,661,685,752,1018,1061,1112,1199"
mask = sha256(ids)
trusted_reference[613].archive_tag XOR mask → b"SIGMOID_DATA_SANITIZATION"
```

## Cum funcționează scripturile (`scripts/`)

Toate primesc ca argument calea către fișierul JSON cu date.

### Trucul comun: testarea rapidă a unei mulțimi candidate
Flag-ul începe mereu cu `SIGMOID_`. Dacă `tag XOR mască = "SIGMOID_..."`, atunci
primii 8 octeți ai măștii = primii 8 octeți ai tag-ului XOR `"SIGMOID_"`. Scripturile
precalculează această valoare pentru **fiecare** înregistrare (dicționarul `pre`). Apoi,
pentru o mulțime candidată, calculează SHA-256 și verifică doar dacă primii 8 octeți
sunt în `pre` – o căutare O(1), fără să încerce XOR cu toate tag-urile.

- **`brute.py`** – pornește de la cele 12 mostre cu caractere ascunse; încearcă toate
  submulțimile lor și apoi „cele 12 + încă o mostră” pentru fiecare mostră rămasă. La
  potrivire afișează mulțimea și textul decodat.
- **`chk.py`** – pentru o mulțime dată, face XOR cu toate tag-urile și afișează
  rezultatele care încep cu `SIGMOID` sau sunt text ASCII lizibil (verificarea finală).
- **`pool.py`, `pool2.py`** – pista intermediară pentru rating-uri: un clasificator
  TF-IDF (uni- și bigrame) + regresie logistică, antrenat pe recenziile de încredere,
  estimează rating-ul probabil al fiecărei mostre; mostrele unde rating-ul stocat e
  improbabil devin candidați, iar combinațiile lor (adăugate la cele 12 de bază, în
  `pool2.py`) sunt testate cu trucul de mai sus. Soluția finală a venit însă din
  potrivirea exactă cu setul Amazon original.

## Rulare

```bash
python3 scripts/chk.py <date.json>
```

## Fișiere
- `solution.md` – rezolvarea.
- `scripts/` – scripturile de căutare și verificare.
- `task.md` – enunțul.
