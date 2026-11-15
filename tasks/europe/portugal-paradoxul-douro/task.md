# Paradoxul Douro (Portugalia)

## Prezentarea sarcinii

### 📖 Povestea
Institutul de viticultură din Valea Douro a testat o nouă metodă de irigare prin picurare, în comparație cu cea veche. Rezultatele par fără echivoc: parcelele irigate prin metoda nouă au succes vizibil mai des. Înainte ca institutul să o recomande pentru toată valea, un stagiar sceptic insistă să împartă datele pe grupuri — și descoperă că, împărțite așa cum trebuie, metoda nouă este mai slabă decât cea veche în absolut fiecare grup.

Treaba ta este să găsești variabila care anulează „victoria”.

### 📌 Titlul sarcinii
Paradoxul Douro

### 🔥 Dificultate
Mediu

### 🎯 Obiectiv
Per ansamblu, `treatment` are o rată `harvest_success` mai mare decât `control`. Găsește singura coloană care, atunci când datele sunt împărțite după ea în grupurile potrivite, face ca `treatment` să fie mai slab decât `control` în fiecare grup — o inversare completă, de tipul paradoxului lui Simpson. Alte câteva coloane par plauzibile, dar nu produc o inversare completă.

### 🚩 FLAG
`SIGMOID_<PHRASE>`, unde `<PHRASE>` este numele coloanei identificate, scris cu MAJUSCULE, cu liniuțele de subliniere păstrate (de ex. `soil_ph` → `SIGMOID_SOIL_PH`).

### 📂 Datele de intrare
`douro_vineyard_data.csv` — 3.543 de rânduri, câte unul pentru fiecare parcelă experimentală:

| Coloană | Descriere |
|---|---|
| `plot_id` | Identificatorul unic al parcelei |
| `treatment` | `control` (metoda veche) sau `treatment` (noua irigare prin picurare) |
| `soil_depth_cm` | Adâncimea solului, în cm |
| `trellis_system` | `vertical shoot positioning`, `lyre`, `bush vine` |
| `vineyard_age_group` | `young (<15y)`, `mature (15-40y)`, `old (>40y)` |
| `sun_hours_july` | Ore de soare în iulie |
| `vine_density_per_ha` | Butuci de viță la hectar |
| `cooperative` | Cooperativa vinicolă de care aparține parcela (`COOP-01` … `COOP-12`) |
| `annual_rainfall_mm` | Precipitații anuale, în mm |
| `grape_variety` | `Touriga Nacional`, `Touriga Franca`, `Tinta Roriz` |
| `distance_from_porto_km` | Distanța rutieră față de Porto, în km |
| `organic_certified` | `yes` / `no` |
| `elevation_m` | Altitudinea, în metri |
| `soil_ph` | pH-ul solului |
| `harvest_month` | `September` sau `October` |
| `slope_aspect` | `north-facing` / `south-facing` / `east-facing` / `west-facing` |
| `harvest_success` | `1` dacă parcela a atins ținta de producție/calitate, altfel `0` |

### ✅ Rezultatul așteptat
- Flagul, `SIGMOID_<PHRASE>`.
- Un URL (GitHub, Google Drive, Colab, ...) către un notebook sau o descriere a soluției care să arate:
  - ratele de succes per ansamblu pentru `treatment` față de `control` (`treatment` în avantaj);
  - cum ai împărțit în grupuri coloana identificată și ratele de succes din fiecare grup (`control` în avantaj în fiecare);
  - o propoziție despre motivul pentru care comasarea grupurilor produce inversarea.

### 🏆 Criterii de evaluare
Această sarcină folosește evaluarea prin flag; organizatorii verifică manual descrierea soluției indicată prin link.
- ✅ Flag corect și o descriere a soluției care arată inversarea în fiecare grup: 200 de puncte (PASS)
- ❌ Flag incorect sau flag corect fără o descriere a soluției care să îl susțină: 0 puncte (FAIL)
