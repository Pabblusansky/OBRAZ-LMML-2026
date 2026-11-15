# Grecia – Doar o chestie între frați

| Câmp | Valoare |
|---|---|
| Țara | Grecia |
| Titlu | Doar o chestie între frați |
| Dificultate | Mediu |
| Categorie | clasificare binară / evaluare modele pe date dezechilibrate |
| Status | rezolvat (`ARES` ales cu `Recall = 95.00%`, `FN = 5`, `F2 = 0.8796`) |
| Flag | `SIGMOID_ARES` |

## Problema pe scurt

Sanctuarul de la Delfi folosește trei modele-oracol PyTorch (**Ares**, **Athena** și **Apollo**) pentru a clasifica întâlnirile cu creaturi mitologice din Grecia antică (`1000` de înregistrări în `test_dataset.csv` și etichetele binare în `test_labels.csv`):
- clasa pozitivă `1` (`is_catastrophic_threat = 1`): **100 de amenințări catastrofale** (10%);
- clasa negativă `0` (`is_catastrophic_threat = 0`): **900 de întâlniri normale** (90%).

Toate cele trei modele raportează exact aceeași **acuratețe (Accuracy = 95,00%)** la pragul de decizie `0.50` (câte 50 de erori din 1000 de exemple), însă în problema descrisă **ratarea unei amenințări catastrofale reale (Fals Negativ, FN)** este mult mai costisitoare decât declanșarea unei **alarme false (Fals Pozitiv, FP)**. Trebuie calculate metricile pe care acuratețea le ascunde și ales modelul potrivit pentru a apăra Delfi.

## De ce acuratețea este înșelătoare (capcana datelor dezechilibrate)

Într-un set de date cu **10% exemple pozitive** și **90% exemple negative**, acuratețea globală $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$ tratează identic o amenințare ratată (`FN`) și o alarmă falsă (`FP`). Mai mult, chiar și un clasificator banal care ar prezice mereu `0` ar obține 90% acuratețe, ratând 100% din amenințări.

Când $FP + FN = 50$ pentru toate cele trei modele, fiecare dintre ele face același număr total de greșeli (`50` din `1000`), dar **distribuția acestor greșeli între `FN` și `FP` este complet diferită**.

## Arhitectura și comportamentul intern al modelelor

Toate cele trei checkpoint-uri (`.pth`) folosesc arhitectura din `model.py` (`MythicThreatClassifier`): un strat liniar PyTorch `nn.Linear(9, 1)` aplicat asupra celor 9 caracteristici numerice din `FEATURE_COLUMNS`:

$$z = \mathbf{w} \cdot \mathbf{x} + b, \qquad P(y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

La pragul de decizie `0.50`, condiția $\sigma(z) \ge 0.50$ este echivalentă cu $z \ge 0$. Inspectarea ponderilor extrase din fișierele `.pth` arată că fiecare oracol se bazează exclusiv pe o singură caracteristică (pondere `1.0` pe o singură coloană, `0.0` pe restul și `bias = 0.0`):

1. **Ares** (`models/model_ares.pth`): pondere `1.0` pe **`oracle_omen`**
2. **Athena** (`models/model_athena.pth`): pondere `1.0` pe **`aegis_resonance`**
3. **Apollo** (`models/model_apollo.pth`): pondere `1.0` pe **`styx_disturbance`**

## Algoritmi și metode de evaluare folosite

Scriptul `main.py` implementează următorul flux de rezolvare:

1. **Alinierea datelor (`load_test_data`)**: îmbină `test_dataset.csv` și `test_labels.csv` după `encounter_id` (validare `1:1`) și extrage matricea de caracteristici în ordinea exactă din `FEATURE_COLUMNS`.
2. **Inferența modelelor (`predict_with_oracle`)**:
   - Încarcă modelul în PyTorch (`MythicThreatClassifier` + `torch.load` + `torch.sigmoid`), dacă `torch` este instalat în mediul de execuție.
   - Include și o cale fallback matematic echivalentă în NumPy (`_extract_linear_params_from_zip`), care citește tensorii `float32` direct din arhiva ZIP `.pth` și aplică transformarea liniară + funcția sigmoid numeric stabilă.
3. **Evaluarea la pragul `0.50` (`compute_metrics`)**:
   - **Matricea de confuzie**: `TP` (amenințări prinse), `FP` (alarme false), `FN` (amenințări ratate), `TN` (întâlniri sigure respinse corect).
   - **Recall / Sensibilitate (`TPR`)** = $\frac{TP}{TP + FN}$ și **Miss Rate (`FNR`)** = $\frac{FN}{TP + FN}$ — metricile critice când `FN` are cost uriaș.
   - **Precision (`PPV`)** = $\frac{TP}{TP + FP}$ și **Specificity (`TNR`)** = $\frac{TN}{TN + FP}$.
   - **Balanced Accuracy** = $\frac{\text{Recall} + \text{Specificity}}{2}$ — medie echilibrată între cele două clase, imună la dezechilibrul 90/10.
   - **F1-Score** = $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ și **F2-Score** ($\beta = 2$, $F_2 = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$) — scorul $F_2$ penalizează fals-negativele de două ori mai sever decât fals-pozitivele.
   - **ROC-AUC** și **PR-AUC (`Average Precision`)** — măsoară capacitatea de separare a probabilităților pe toate pragurile posibile.
4. **Selecția modelului și generarea dovezilor**: exportă tabelul metricilor (`metrics_comparison.csv`), graficele comparative (`evaluation_comparison.png`) și flag-ul (`flag.txt`).

## Rezultate obținute pe setul de test (prag = `0.50`)

| Model | Caracteristică activă | TP | FP | FN (ratate) | TN | Accuracy | Recall (TPR) | Miss Rate (FNR) | Precision | Specificity (TNR) | Bal. Acc | F1 | F2 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ARES** | `oracle_omen` | **95** | 45 | **5** | 855 | 0.9500 | **0.9500** | **0.0500** | 0.6786 | 0.9500 | **0.9500** | **0.7917** | **0.8796** | **0.9471** | **0.6970** |
| **ATHENA** | `aegis_resonance` | 75 | 25 | 25 | 875 | 0.9500 | 0.7500 | 0.2500 | 0.7500 | 0.9722 | 0.8611 | 0.7500 | 0.7500 | 0.8766 | 0.6389 |
| **APOLLO** | `styx_disturbance` | 55 | 5 | 45 | 895 | 0.9500 | 0.5500 | 0.4500 | 0.9167 | 0.9944 | 0.7722 | 0.6875 | 0.5978 | 0.7859 | 0.6105 |

### De ce este ales **ARES** (`SIGMOID_ARES`)?
- **Apollo** are cea mai mare precizie (`91.67%`) și doar `5` alarme false, dar ratează **45% din amenințările catastrofale (`FN = 45`)** — aproape jumătate din atacuri ar lovi Delfi neanunțate.
- **Athena** oferă un compromis simetric (`25 FP` și `25 FN`), dar tot ratează **25% din amenințări**.
- **Ares** detectează **95 din 100 de amenințări reale (`Recall = 95,00%`, doar `FN = 5`)**, obținând totodată cele mai mari scoruri la **Balanced Accuracy (`0.9500`)**, **F1 (`0.7917`)**, **F2 (`0.8796`)**, **ROC-AUC (`0.9471`)** și **PR-AUC (`0.6970`)**. Deoarece alarmele false (`45 FP`) sunt ieftine față de o catastrofă ratată, **Ares** este oracolul potrivit.

## Rulare

```bash
python3 main.py
```

## Fișiere
- `main.py` – scriptul de evaluare, selecție și generare a materialelor doveditoare (comentat în română).
- `model.py` – definiția arhitecturii `MythicThreatClassifier` și lista `FEATURE_COLUMNS` (comentat în română).
- `models/` – ponderile PyTorch (`model_ares.pth`, `model_athena.pth`, `model_apollo.pth`).
- `test_dataset.csv`, `test_labels.csv` – datele de test și etichetele adevărate.
- `metrics_comparison.csv` – tabelul complet cu metricile calculate pentru cele trei modele.
- `evaluation_comparison.png` – vizualizarea matricilor de confuzie, a metricilor și a curbelor ROC / Precision-Recall.
- `flag.txt` – flag-ul rezultat (`SIGMOID_ARES`).
- `task.md`, `requirements.txt`, `README.MD` – enunțul, dependențele și documentația.
