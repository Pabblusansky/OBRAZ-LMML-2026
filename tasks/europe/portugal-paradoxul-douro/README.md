# Portugalia – Paradoxul Douro

| Câmp | Valoare |
|---|---|
| Țara | Portugalia |
| Dificultate | Mediu |
| Categorie | analiză de date / Paradoxul lui Simpson (variabilă de confuzie) |
| Coloana identificată | `distance_from_porto_km` |
| Flag | `SIGMOID_DISTANCE_FROM_PORTO_KM` |

## Descrierea problemei

Institutul de viticultură din Valea Douro a evaluat o metodă nouă de irigare prin picurare (`treatment`) comparativ cu metoda veche (`control`) pe `3.543` de parcele experimentale din [douro_vineyard_data.csv](douro_vineyard_data.csv).

La nivel global (toate parcelele comasate), `treatment` pare superior față de `control`:
- **`treatment`**: `585 / 1.890 = 30,95%` (`0.309524`)
- **`control`**: `447 / 1.653 = 27,04%` (`0.270417`)
- Diferență aparentă: **+3,91 puncte procentuale** în favoarea noii irigări.

Obiectivul este găsirea **singurei** coloane din setul de date care, atunci când parcelele sunt împărțite pe grupuri corespunzătoare după ea, produce o **inversare completă de tip Paradoxul lui Simpson** — adică face ca `control` să aibă o rată `harvest_success` strict mai mare decât `treatment` în **absolut fiecare grup**.

## Pașii pentru a rezolva problema

1. **Verificarea ratei globale (`harvest_success`):**
   Se grupează datele după `treatment` și se confirmă avantajul agregat al metodei noi (`30,95%` vs. `27,04%`).
2. **Auditul coloanelor categorice:**
   Se calculează ratele `control` vs. `treatment` în fiecare categorie pentru cele 7 coloane categorice (`trellis_system`, `vineyard_age_group`, `cooperative`, `grape_variety`, `organic_certified`, `harvest_month`, `slope_aspect`). Se observă că cel mult câteva categorii izolate inversează (`3/12` la `cooperative`, `1/4` la `slope_aspect`, `0` la restul), deci niciuna nu produce o inversare completă.
3. **Analiza corelației și a distribuției coloanelor numerice:**
   Se verifică corelația fiecărei coloane numerice cu alocarea la `treatment` și cu `harvest_success`. Singura coloană puternic corelată cu alocarea tratamentului este `distance_from_porto_km` (`r = +0,5789`), în timp ce toate celelalte coloane numerice au corelație aproape zero cu alocarea (`|r| < 0,03`).
4. **Căutarea exhaustivă prin Programare Dinamică (DP):**
   Pentru a exclude orice pistă falsă indiferent de modul de împărțire în intervale, se rulează un algoritm DP (implementat și explicat în română în [main.py](main.py)) peste valorile sortate ale fiecărei coloane numerice. Algoritmul demonstrează că:
   - Pentru `soil_depth_cm`, `sun_hours_july`, `vine_density_per_ha`, `annual_rainfall_mm`, `elevation_m` și `soil_ph` **nu există nicio partiționare** în ≥ 2 intervale contigue în care `control > treatment` în toate grupurile (nici măcar `soil_depth_cm`, care este o pistă falsă cu gol între `70` și `110 cm`).
   - Singura coloană care permite inversarea completă este **`distance_from_porto_km`**.
5. **Identificarea golului natural și împărțirea în grupuri pe `distance_from_porto_km`:**
   Distribuția valorilor lui `distance_from_porto_km` (`70,0 – 204,9 km`) are un **gol natural de 10 km** fără nicio parcelă între **104,0 km** și **114,0 km**. Împărțind datele în acest punct (prag `110 km`), se obține inversarea completă:

   | Grup (`distance_from_porto_km`) | `control` | `treatment` | Diferență (`T − C`) |
   |---|---:|---:|---:|
   | **Aproape de Porto (`70,0 – 104,0 km`)** | 312 / 1.233 (**25,30%**) | 36 / 144 (**25,00%**) | **−0,30 pp** (`control` câștigă) |
   | **Departe de Porto (`114,0 – 204,9 km`)** | 135 / 420 (**32,14%**) | 549 / 1.746 (**31,44%**) | **−0,70 pp** (`control` câștigă) |

   *(Inversarea rămâne valabilă în toate grupurile și la împărțirea în 3 subregiuni viticole: `70–110 km`: `25,30% vs. 25,00%`, `110–145 km`: `34,33% vs. 32,62%`, `145–205 km`: `31,12% vs. 30,81%`.)*
6. **Explicarea mecanismului paradoxului:**
   Comasarea grupurilor produce inversarea deoarece `92,38%` (`1.746 / 1.890`) dintre parcelele `treatment` sunt situate departe de Porto (`≥ 114 km`), unde rata de succes de bază este mai ridicată (`~31–32%`), pe când `74,59%` (`1.233 / 1.653`) dintre parcelele `control` sunt situate aproape de Porto (`≤ 104 km`), unde rata de succes de bază este mai mică (`~25%`), distorsionând astfel media agregată în favoarea noii metode.

## Rulare

```bash
python3 main.py
```

Scriptul tipărește analiza completă în terminal și salvează raportul în [solution.txt](solution.txt).

## Fișiere

| Fișier | Rol |
|---|---|
| [task.md](task.md) | Enunțul complet al sarcinii |
| [main.py](main.py) | Scriptul Python de analiză și demonstrație (comentat integral în română) |
| [solution.ipynb](solution.ipynb) | Notebook Jupyter executat (cu tabele și grafice salvate, pregătit pentru Colab / Google Drive / GitHub) |
| [solution.md](solution.md) | Descrierea concisă a soluției pentru evaluarea manuală |
| [solution.txt](solution.txt) | Raportul text generat automat de `main.py` |
| [douro_vineyard_data.csv](douro_vineyard_data.csv) | Setul de date experimental (copiat și în [files/douro_vineyard_data.csv](files/douro_vineyard_data.csv)) |
