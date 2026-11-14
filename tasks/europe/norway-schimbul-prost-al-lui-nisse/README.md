# Schimbul prost al lui nisse (Norway)

## Analiza problemei
Setul de date conține imagini 8×8 de cifre scrise de mână (`0–9`, valori `0–16`) provenite din `sklearn.datasets.load_digits()` (1,797 eșantioane în total: `1,198` în `train-input.csv` și `599` în `test-input.csv`).

În nopțile în care nisse nu a primit untul din terci, acesta a inversat sistematic între ele **două cifre specifice** pe întreaga durată a schimbului respectiv:
- **`train-input.csv`**: schimb prost între indicii `452..596` (`IMG_0452`..`IMG_0596`), unde cifrele **`4` și `9`** au fost inversate (`36` etichete corupte, `0` în afara intervalului).
- **`test-input.csv`**: schimb prost între indicii `162..298` (`IMG_1360`..`IMG_1496`, în cadrul blocului `[150, 300)`), unde cifrele **`3` și `8`** au fost inversate sistematic pe **`30` de rânduri**.

## Abordare (`main.py`)
1. Antrenează un clasificator `StandardScaler + SVC(C=10, gamma="scale", random_state=42)` pe etichetele curate din `train-output.csv`.
2. Aplică algoritmul lui Kadane peste toate perechile de cifre `(a, b)` comparând `given_label` cu predicțiile clasificatorului pe `test-input.csv`, detectând intervalul contiguu `IMG_1360..IMG_1496` cu perechea `(3, 8)` (scor Kadane `29.0`).
3. Aplică inversarea `3 <-> 8` exclusiv pe etichetele din intervalul detectat și validează reparațiile față de `sklearn.datasets.load_digits()` (`Precision = 1.0000`, `Recall = 1.0000`, `F1 = 1.0000`, `30 / 30` etichete reparate).
4. Generează `test-output.csv` (`sample_id,corrected_label`, `599` rânduri) și reantrenează clasificatorul pe întregul set de etichete corectate (`1,797` exemple, acuratețe 5-fold CV `0.9822`, acuratețe pe setul corectat `1.0000`), salvându-l în `model.joblib`.

## Rulare și verificare
```bash
/Users/dragoschisca/Desktop/LMML/.venv/bin/python main.py
```

### Fișiere generate
- `test-output.csv` — predicțiile corectate pentru `test-input.csv` (`599` rânduri, `IMG_1198..IMG_1796`)
- `model.joblib` — clasificatorul `Pipeline(StandardScaler, SVC)` reantrenat pe etichetele corectate
