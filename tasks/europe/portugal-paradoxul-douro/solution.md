# Soluție – Paradoxul Douro (Portugalia)

**Flag:** `SIGMOID_DISTANCE_FROM_PORTO_KM`  
**Coloana identificată:** `distance_from_porto_km`

---

## 1. Ratele de succes per ansamblu (`treatment` vs. `control`)

Pe întregul set de date ([douro_vineyard_data.csv](douro_vineyard_data.csv), `3.543` parcele experimentale), fără nicio împărțire pe grupuri, noua metodă de irigare prin picurare (`treatment`) pare câștigătoare cu **+3,91 puncte procentuale**:

| Metodă | Parcele | Succese (`harvest_success = 1`) | Rata de succes |
|---|---:|---:|---:|
| **`treatment`** (noua irigare prin picurare) | 1.890 | 585 | **30,95%** (`0.309524`) |
| **`control`** (metoda veche) | 1.653 | 447 | **27,04%** (`0.270417`) |
| **Diferență (`treatment − control`)** | — | — | **+3,91 pp** |

---

## 2. Cum am împărțit coloana `distance_from_porto_km` și ratele din fiecare grup

Inspecția distribuției coloanei `distance_from_porto_km` (`70,0 – 204,9 km`) evidențiază un **gol geografic natural de 10 km** între **104,0 km** și **114,0 km** (zero parcele în intervalul `(104,0; 114,0)`), care împarte experimental valea în două regimuri distincte — aproape de Porto vs. departe de Porto.

### Varianta A: Împărțirea naturală în 2 grupuri (prag la golul natural `110 km`, i.e. `≤ 104 km` vs. `≥ 114 km`)

| Grup (`distance_from_porto_km`) | `control` (succ/n → rată) | `treatment` (succ/n → rată) | Diferență (`treatment − control`) | Avantaj |
|---|---:|---:|---:|---|
| **Aproape de Porto (`70,0 – 104,0 km`)** | 312 / 1.233 → **25,30%** (`0.253041`) | 36 / 144 → **25,00%** (`0.250000`) | **−0,30 pp** | **`control`** |
| **Departe de Porto (`114,0 – 204,9 km`)** | 135 / 420 → **32,14%** (`0.321429`) | 549 / 1.746 → **31,44%** (`0.314433`) | **−0,70 pp** | **`control`** |

### Varianta B: Împărțirea în 3 subregiuni viticole din Valea Douro (`70–110 km`, `110–145 km`, `145–205 km`)

Inversarea completă se păstrează și dacă împărțim zona îndepărtată în subregiuni corespunzătoare zonelor viticole ale văii:

| Subregiune (`distance_from_porto_km`) | `control` (succ/n → rată) | `treatment` (succ/n → rată) | Diferență (`treatment − control`) | Avantaj |
|---|---:|---:|---:|---|
| **Baixo Corgo (`70 – 110 km`)** | 312 / 1.233 → **25,30%** | 36 / 144 → **25,00%** | **−0,30 pp** | **`control`** |
| **Cima Corgo (`110 – 145 km`)** | 46 / 134 → **34,33%** | 199 / 610 → **32,62%** | **−1,71 pp** | **`control`** |
| **Douro Superior (`145 – 205 km`)** | 89 / 286 → **31,12%** | 350 / 1.136 → **30,81%** | **−0,31 pp** | **`control`** |

În absolut **fiecare** grup, metoda veche (`control`) are o rată de succes mai mare decât noua irigare prin picurare (`treatment`).

---

## 3. De ce comasarea grupurilor produce inversarea (Paradoxul lui Simpson)

Comasarea grupurilor produce inversarea deoarece `distance_from_porto_km` acționează ca o variabilă de confuzie puternică atât asupra alocării tratamentului, cât și asupra ratei de succes de bază: **92,38%** (`1.746` din `1.890`) dintre parcelele `treatment` au fost amplasate departe de Porto (`≥ 114 km`), unde succesul de bază este ridicat (`~31–32%`), în timp ce **74,59%** (`1.233` din `1.653`) dintre parcelele `control` au rămas aproape de Porto (`≤ 104 km`), unde succesul de bază este scăzut (`~25%`), făcând ca media agregată a metodei noi să reflecte avantajul geografic, nu eficacitatea irigării.

---

## 4. De ce celelalte coloane sunt piste false

Verificarea exhaustivă din [main.py](main.py) (și din notebook-ul executat [solution.ipynb](solution.ipynb)) arată că **nicio altă coloană** nu produce o inversare completă:

- **Coloane categorice:**
  - `cooperative`: inversează doar în **3/12** cooperative (`COOP-03`, `COOP-04`, `COOP-07`).
  - `slope_aspect`: inversează doar în **1/4** expoziții (`west-facing`).
  - `trellis_system` (`0/3`), `vineyard_age_group` (`0/3`), `grape_variety` (`0/3`), `organic_certified` (`0/2`), `harvest_month` (`0/2`): `treatment` rămâne superior în **toate** grupurile.
- **Coloane numerice (demonstrație prin Programare Dinamică pe intervale contigue):**
  - Chiar testând *toate* partițiile posibile în ≥ 2 intervale contigue (atât cu minim `20` parcele per braț, cât și cu minim `1` parcelă per braț), `soil_depth_cm`, `sun_hours_july`, `vine_density_per_ha`, `annual_rainfall_mm`, `elevation_m` și `soil_ph` nu admit **nicio** împărțire în care `control > treatment` peste tot (`max_intervale = -1`).
  - În particular, `soil_depth_cm` imită o distribuție bimodală (gol între `70 cm` și `110 cm`), dar împărțită în sol superficial (`40–70 cm`) vs. sol adânc (`110–160 cm`) păstrează `treatment > control` în ambele grupuri (`33,75% vs. 27,85%`, respectiv `28,50% vs. 26,41%`).
