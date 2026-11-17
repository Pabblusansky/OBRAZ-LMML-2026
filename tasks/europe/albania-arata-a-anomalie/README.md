# Albania – Arată a anomalie!

| Câmp | Valoare |
|---|---|
| Țara | Albania |
| Dificultate | Mediu |
| Categorie | detecție de anomalii în flux (streaming) pe serii de timp |
| Status | rezolvat |
| Flag | `SIGMOID_<WORD>` (eliberat la scor ≥ 70) |

## Problema pe scurt

Senzorii din buncărele albaneze trimit câte o citire pe minut: `temperature_c`,
`humidity_pct`, `vibration_g`, `acoustic_db`, `air_pressure_kpa`, `co2_ppm`.
Trebuie implementată clasa `Detector` cu metoda `update(reading) -> float`, care
primește **câte o citire pe rând** (nu vede viitorul) și întoarce un scor în [0, 1];
peste 0,50 citirea e declarată anomalie.

Există trei tipuri de anomalii:
- **point** – un vârf izolat pe o singură citire;
- **drift** – o deplasare lentă a nivelului, pe mai multe minute;
- **collective** – un grup de citiri anormale împreună, adesea pe mai mulți senzori.

Scor = 100 × (0,80·F1 + 0,05·recall point + 0,075·recall drift + 0,075·recall collective).
Limite: Python 3.11, numpy/pandas/scikit-learn, CPU, 60 s, fără internet.

## Ideea algoritmului

Detectorul compară ce vede **acum** cu un **„normal” de referință** construit din trecut,
și măsoară diferența în unități de zgomot (scor z). Trei detectoare lucrează în paralel,
iar scorul final e maximul lor.

```
citire → curățare vârfuri → ferestre recente (3 și 6 min)
                         ↘ referință = mediana a 90 de citiri normale, întârziată cu 30 min
       scor z = (fereastră − referință) / σ_zgomot   → sigmoidă → max → scor
```

### 1. Detectorul de vârfuri (anomalii *point*)
- Mediana locală a ultimelor 5 citiri brute = ce „ar trebui” să fie valoarea.
- `dev_point = citire − mediana locală`, împărțit la σ al acestei diferențe → `z_point`.
- Dacă `|zp| > 7` pe vreun canal → vârf. Valoarea e **înlocuită cu mediana** în
  fluxul „curățat”, ca vârful să nu strice ferestrele și referința.

### 2. Referința întârziată (baseline)
- Mediana (robustă la valori extreme) a ultimelor `BASE_N = 90` citiri considerate normale.
- Citirile intră în referință abia după `LAG + W_LONG = 36` de pași (coada `pending`) și
  doar dacă nu au fost semnalate ca anomalii. Astfel un drift lent nu e „înghițit” de
  referință înainte să fie detectat.

### 3. Detectoarele de nivel (anomalii *drift* și *collective*)
- `dev_long` = media ferestrei lungi (6 citiri) − referința; `dev_short` = media ferestrei scurte (3).
- Împărțite la σ-urile lor → `z_long`, `z_short`.
- Patru scoruri, fiecare trecut printr-o sigmoidă `σ(2·(z − prag))`:
  - fereastra lungă, cel mai mare canal, prag 3,5 (drift);
  - fereastra scurtă, cel mai mare canal, prag 5,0 (salt brusc);
  - fereastra lungă, **al doilea** canal ca mărime, prag 3,0 (≥ 2 senzori de acord →
    colectiv, chiar dacă fiecare e doar moderat);
  - fereastra scurtă, al doilea canal, prag 4,0.
- Scorul final = maximul celor cinci scoruri (vârf + patru de nivel).

### 4. Estimarea zgomotului (σ)
- **Încălzire**: primele 150 de citiri întorc scor 0 și doar colectează diferențe; la
  final σ inițial se estimează robust cu MAD (mediana abaterilor absolute × 1,4826).
- Apoi σ² se actualizează lent (medie exponențială cu `α = 0,005`), cu contribuția
  plafonată la 16·σ² ca un outlier să nu umfle zgomotul; când citirea e semnalată
  anomalie, actualizarea e atenuată de 20 de ori.

### 5. Reguli de final de alarmă
- Dacă alarma a durat ≥ 10 pași, dar fereastra scurtă a revenit la normal (`z < 2`),
  scorul e coborât la 0,3: fereastra lungă încă „ține minte” anomalia, dar ea s-a terminat.
- Dacă o alarmă ține peste 150 de pași, citirile sunt acceptate în referință (se presupune
  un nou regim normal, ca detectorul să nu rămână blocat).

### Robustețe
Valorile lipsă sau ne-numerice sunt înlocuite cu ultima valoare curată; orice excepție
dă scor 0; scorul e mereu limitat la [0, 1].

## Rulare și evaluare locală

```bash
cd work && python evaluate.py ../detector.py
```

`work/evaluate.py` rulează detectorul pe `files/train-input.csv`, calculează F1, recall
pe tipuri și scorul oficial (cu `-v` listează citirile greșite). `work/sweep.py` face
căutare pe grilă a constantelor (praguri, ferestre). Versiunile intermediare
ale detectorului au fost șterse; singura versiune e `detector.py`.


## Fișiere
- `detector.py` – soluția trimisă.
- `work/` – scripturile de evaluare (`evaluate.py`) și de căutare a parametrilor (`sweep.py`).
- `files/` – datele primite (train-input.csv, train-output.csv).
