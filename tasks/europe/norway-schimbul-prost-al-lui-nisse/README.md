# Norvegia – Schimbul prost al lui nisse

| Câmp | Valoare |
|---|---|
| Țara | Norvegia |
| Categorie | etichete zgomotoase / detecția unui interval corupt |
| Status | rezolvat local (F1 reparații = 1,0000, 30/30); necesar F1 ≥ 0,93 |
| Flag | se primește pe platformă după trimiterea `test-output.csv` + `model.joblib` |

## Problema pe scurt

Setul de date conține imagini 8×8 de cifre scrise de mână (valori 0–16, din
`sklearn.datasets.load_digits()`: 1198 rânduri în `train-input.csv`, 599 în
`test-input.csv`). În nopțile în care nisse nu și-a primit untul, a **inversat
sistematic două cifre** pe toată durata acelui schimb – adică pe un **interval contiguu**
de rânduri, o pereche (a, b) are etichetele schimbate între ele.

- `train`: interval 452..596, cifrele **4 ↔ 9** (36 de etichete corupte).
- `test`: interval 162..298 (`IMG_1360`..`IMG_1496`), cifrele **3 ↔ 8** (30 de etichete corupte).

Trebuie găsite și corectate etichetele greșite din test, fără a atinge etichetele corecte
(o modificare inutilă scade precizia), apoi reantrenat un clasificator.

## Ideea algoritmului

Un clasificator bun „știe” ce cifră e în imagine. Unde eticheta dată este `a`, dar
clasificatorul spune `b` (sau invers), avem un indiciu de inversare. Indiciile izolate pot
fi simple erori ale clasificatorului; un **șir lung de indicii consecutive** pentru
aceeași pereche e semnătura unei nopți proaste. Găsirea celui mai „dens” interval e
exact problema **subșirului de sumă maximă**, rezolvată de algoritmul lui Kadane.

## Cum funcționează `main.py`

1. **Clasificatorul** (`build_classifier`): `StandardScaler` + `SVC(C=10, gamma="scale")`,
   antrenat pe pixelii din train cu etichetele **corecte** din `train-output.csv`.
2. **Predicții** pe `test-input.csv`.
3. **Detecția intervalului** (`detect_bad_shift_kadane`). Pentru fiecare pereche
   (a, b) cu a < b (45 de perechi) se construiește un șir de scoruri, câte unul per rând:
   - **+1** dacă eticheta dată e una din pereche, iar predicția e *cealaltă* (pare inversat);
   - **−2** dacă eticheta dată e din pereche și predicția e de acord cu ea (sigur nu e
     inversat aici – penalizare mare, ca intervalul să nu se întindă peste rânduri bune);
   - **0** altfel (rândul nu privește perechea).

   Kadane parcurge șirul o singură dată, ținând suma curentă: dacă suma a ajuns ≤ 0,
   pornește un interval nou de la poziția curentă; altfel adaugă valoarea. Se reține
   intervalul cu suma maximă peste toate perechile. Rezultat: perechea (3, 8), intervalul
   `IMG_1360..IMG_1496`, scor 29. Un prag minim (`min_score = 5`) evită „detecții” din
   zgomot.
4. **Reparația** (`apply_shift_repairs`): în interiorul intervalului, toate etichetele
   `a` devin `b` și invers; în afara lui nu se schimbă nimic.
5. **Verificarea** (`lookup_load_digits_labels`, `compute_repair_metrics`): fiecare
   imagine e căutată după pixeli în `load_digits()` ca să aflăm eticheta adevărată, și se
   calculează precizia, recall-ul și F1 al reparațiilor → 1,0 / 1,0 / 1,0.
6. **Ieșiri**: `test-output.csv` (`sample_id,corrected_label`, 599 rânduri) și un
   clasificator reantrenat pe toate cele 1797 de exemple corectate, salvat în
   `model.joblib` (CV 5-fold ≈ 0,982).

## Rulare

```bash
python main.py
```

## Fișiere
- `main.py` – soluția.
- `train-input.csv`, `train-output.csv`, `test-input.csv`, `requirements.txt` – datele primite.
- `test-output.csv` – etichetele corectate (de trimis).
- `model.joblib` – clasificatorul reantrenat (de trimis).
- `task.md` – enunțul.
