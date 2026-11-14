# Belgia – Fry-ndship

| Câmp | Valoare |
|---|---|
| Țara | Belgia (Flandra) |
| Dificultate | Ușor |
| Categorie | regresie liniară / diagnosticarea reziduurilor |
| Date | `potato_fields.csv` (3869 parcele) |
| Flag-uri găsite | `SIGMOID_REGRESSION` (stratul 1), `SIGMOID_SUPERVISION` (stratul 2), `SIGMOID_PREDICTION` (stratul 3) |

## Problema pe scurt

Trebuie modelată lungimea marginii parcelelor de cartofi (`edge_m`) din dimensiunile lor
și explicat de ce măsurătorile Mariettei par „ciudat de organizate”, deși o regresie
liniară naivă explică deja ~99% din varianță (R² ≈ 0,989).

Răspunsul: mesajele sunt **ascunse în reziduuri** (diferența real − prezis). Ele apar
doar când modelul are forma corectă; fiecare variabilă în plus „decojește” un strat și
dezvăluie alt cuvânt.

## Ideea matematică

Fiecare parcelă e un dreptunghi cu aria A = 10000 · `area_ha` (m²) și latura lungă
L = `long_side_m`. Derivăm latura scurtă și perimetrul:

```
W = short_side_m = 10000 · area_ha / L
P = 2 · (L + W)
```

De ce modelul naiv ratează mesajul: W ∝ A / L este **neliniar** în (A, L). O regresie
`edge_m ~ area_ha + long_side_m` nu poate reprezenta împărțirea, deci reziduurile au o
curbură parabolică mare (de la −80 la +105 m) care acoperă complet literele.
Odată ce dăm modelului direct W, relația devine liniară și rămâne doar mesajul.

## Cele patru modele (`main.py`)

| Model | Trăsături | Ecuația găsită | R² | RMSE | Reziduul arată |
|---|---|---|---|---|---|
| Naiv | `area_ha`, `long_side_m` | 140,95 + 60,71·area + 1,44·L | 0,98901 | 25,34 m | curbură, nimic lizibil |
| Strat 1 | `L`, `W` | ≈ 1,90·L + 1,90·W = 0,95·P | 0,99837 | 9,77 m | **REGRESSION** |
| Strat 2 | `L`, `W`, `ditch_m` | ≈ 0,45·P + ditch | 0,9999843 | 0,96 m | **SUPERVISION** |
| Strat 3 | `L`, `W`, `ditch_m`, `hedge_m` | ≈ 0,41·P + ditch + hedge | 0,99999994 | 0,058 m | **PREDICTION** |

Datele sunt construite ca reziduurile fiecărui strat să conțină:
- o **bandă interioară** de puncte care desenează literele (ex. |e₁| < 12,5 m);
- **benzi exterioare de compensare** care țin media reziduurilor și corelația lor cu
  trăsăturile la zero – exact condițiile pe care OLS le impune, ca mesajul să
  „supraviețuiască” ajustării.

## Cum funcționează `main.py`, pas cu pas

1. `load_and_engineer_features` – citește CSV-ul și calculează `short_side_m` (W) și
   `perim_rect_m` (P).
2. `ModelSpec` – descrie fiecare model: trăsăturile, banda interioară de evidențiat,
   flag-ul și numele imaginii.
3. `fit_model` – regresie liniară OLS (`sklearn.LinearRegression`), apoi predicții,
   reziduuri, R² și RMSE.
4. `format_equation` – afișează ecuația ajustată cu 4 zecimale.
5. `plot_residual_axis` – graficul reziduu vs valoarea prezisă ŷ; punctele din banda
   interioară sunt negre (literele), cele exterioare gri deschis. Se folosește ŷ pe axa X
   (nu P), pentru că straturile 2 și 3 includ decalaje din `ditch_m` / `hedge_m`.
6. `save_individual_plot` – imagine lată (15 × 3,8) per strat, ca literele să nu fie
   comprimate; `save_combined_figure` – toate cele 4 panouri într-o figură.
7. `main` – rulează cele 4 modele, afișează metricile și salvează imaginile.

## Rulare

```bash
python main.py
```

## Fișiere

| Fișier | Rol |
|---|---|
| `potato_fields.csv` | datele de intrare (3869 parcele) |
| `main.py` | trăsăturile derivate, regresiile OLS și graficele de reziduuri |
| `residuals_fry_ndship.png` | figura cu 4 panouri (naiv + cele 3 straturi) |
| `layer1_regression.png` | reziduurile stratului 1 → `SIGMOID_REGRESSION` |
| `layer2_supervision.png` | reziduurile stratului 2 → `SIGMOID_SUPERVISION` |
| `layer3_prediction.png` | reziduurile stratului 3 → `SIGMOID_PREDICTION` |
