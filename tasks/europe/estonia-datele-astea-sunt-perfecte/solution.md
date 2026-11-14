# Soluție – Datele astea sunt perfecte! (Estonia)

**Flag:** `SIGMOID_76` *(sau `SIGMOID_77` cu rotunjire la cel mai apropiat întreg; vezi detalii mai jos)*  
**Acuratețe reală:** **`0.7666`** (`76.66%`, `1511 / 1971` cu `LeaveOneGroupOut` pe toate rândurile) / **`0.7656`** (`76.56%`, `689 / 900` cu `LeaveOneOut` după deduplicarea aplicanților)

---

## 1. Bugul adevărat vs. pistele false

La inspecția [project_pipeline.py](files/project_pipeline.py) și a datelor din [applications.csv](files/applications.csv) apar mai multe alegeri discutabile. Doar **una** este responsabilă pentru inflația falsă de la ~76.6% la **93.42%**:

### Bugul adevărat: Data Leakage la nivel de aplicant / entitate între `train` și `test`
Pipeline-ul original elimină coloanele de identitate (`submission_id`, `date_of_birth`, `citizenship`) înainte de împărțirea datelor:
```python
X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"])
```
Problema este că cele **1,971 de rânduri** din `applications.csv` nu sunt 1,971 de aplicanți independenți, ci provin de la doar **900 de persoane unice** identificate prin perechea `(date_of_birth, citizenship)`:
- Distribuția numărului de cereri per aplicant: **360** cu 1 cerere, **195** cu 2 cereri, **159** cu 3 cereri, **186** cu 4 cereri.
- Pentru fiecare aplicant cu mai multe cereri, `label` este **100% identic** (0 grupuri cu etichete diferite), iar cele 6 caracteristici numerice sunt variații aproape identice ale aceleiași cereri (deviația standard intra-grup este ~`0.87%–1.00%` din deviația totală; exemplu: grupul `('1960-02-16', 'FR')`, rândurile `SUB-000251`, `SUB-001737`, `SUB-001852`, `SUB-001920`).
- În spațiul scalat cu `StandardScaler`, **distanța euclidiană maximă între cereri ale aceluiași aplicant este `0.1018`**, strict mai mică decât **distanța minimă dintre doi aplicanți diferiți (`0.2812`)**.

Prin urmare, un `train_test_split` aleatoriu pe rânduri pune cereri ale aceleiași persoane atât în `X_train`, cât și în `X_test`:
- În split-ul original (`random_state=1137`, `395` rânduri de test), **308 rânduri (77.97%)** au același aplicant prezent și în `X_train` — pe acestea `KNeighborsClassifier(n_neighbors=1)` obține **1.0000 (100.00%)**.
- Pe cele **87 de rânduri (22.03%)** ale aplicanților nevăzuți în `X_train`, modelul obține doar **0.7011 (70.11%)**.
- Scorul raportat (`0.9342`) este media ponderată a memorării aplicanților duplicați și a generalizării reale: `(308 × 1.0000 + 87 × 0.7011) / 395 = 0.9342`.

---

## 2. Dovezi pentru pistele false (red herrings)

Fiecare pistă falsă a fost testată separat în [corrected_pipeline.py](corrected_pipeline.py). Repararea lor nu schimbă performanța decât în limitele zgomotului normal de la o rulare la alta:

| Variantă testată | Descriere | Acuratețe (`rs=1137`) | Alte verificări / CV |
|---|---|---|---|
| **Pipeline original** | Codul livrat (`project_pipeline.py`) | **0.9342** (`369/395`) | `LeaveOneOut()` naiv pe 1,971 rânduri: `0.9594` |
| **Pista falsă A: Leakage în `StandardScaler`** | `StandardScaler().fit_transform(X)` rulat pe tot setul înainte de `train_test_split` → mutat în `make_pipeline` (fit doar pe `X_train`) | **0.9367** (`93.67%`) | Medie pe 100 seed-uri: `0.9440 ± 0.0113` (vs `0.9443 ± 0.0112` cu leakage); `LeaveOneOut()`: `0.9594` |
| **Pista falsă B: Valori imposibile / necurățate** | 41 de rânduri au valori fizic imposibile (`form_fill_seconds < 0`: 9, `email_domain_age_days < 0`: 6, `ip_geo_distance_km < 0`: 10, `address_history_years < 0`: 9, `doc_scan_quality > 1.0`: 7) → testat prin eliminare (B1) sau clipping în `[0, ∞)` / `[0, 1]` (B2) | **0.9430** (B1 drop)<br>**0.9316** (B2 clip) | `LeaveOneOut()` pe 1,930 rânduri curate: `0.9611`<br>`LeaveOneOut()` cu clipping: `0.9589` |
| **Pista falsă C: Split nestratificat / un singur `random_state`** | Adăugare `stratify=y` în `train_test_split`, `StratifiedKFold(5)`, sau `LeaveOneOut()` pe toate rândurile | **0.9418** (`stratify=y`) | `StratifiedKFold(5)`: `0.9432`; `LeaveOneOut()` naiv: `0.9594` |

Niciuna dintre aceste corecții nu elimină scurgerea de entități între `train` și `test`, motiv pentru care acuratețea rămâne fals umflată în zona `93%–96%`.

---

## 3. Pipeline-ul corectat

Păstrăm modelul impus (`StandardScaler` + `KNeighborsClassifier(n_neighbors=1)` pe cele 6 caracteristici numerice) și eliminăm scurgerea de aplicanți folosind validarea leave-one-group-out (`LeaveOneGroupOut()` pe grupuri `(date_of_birth, citizenship)`) și, echivalent, `LeaveOneOut()` pe setul deduplicat la câte o cerere per aplicant (`900` rânduri):

```python
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("applications.csv")
feature_cols = [
    "form_fill_seconds",
    "doc_scan_quality",
    "photo_match_score",
    "email_domain_age_days",
    "ip_geo_distance_km",
    "address_history_years",
]
X = df[feature_cols]
y = df["label"]
groups = df.groupby(["date_of_birth", "citizenship"]).ngroup()

pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))

# 1. Echivalentul leave-one-out la nivel de aplicant pe toate cele 1971 de rânduri:
preds_logo = cross_val_predict(pipe, X, y, cv=LeaveOneGroupOut(), groups=groups)
acc_logo = accuracy_score(y, preds_logo)
print(f"LeaveOneGroupOut accuracy: {acc_logo:.4f}")  # 0.7666 (1511 / 1971)

# 2. Alternativ: LeaveOneOut() după deduplicare la un rând per (date_of_birth, citizenship):
df_dedup = df.drop_duplicates(subset=["date_of_birth", "citizenship"], keep="first")
preds_loo = cross_val_predict(pipe, df_dedup[feature_cols], df_dedup["label"], cv=LeaveOneOut())
acc_loo = accuracy_score(df_dedup["label"], preds_loo)
print(f"Deduplicated LeaveOneOut accuracy: {acc_loo:.4f}")  # 0.7656 (689 / 900)
```

Scriptul complet de verificare este în [corrected_pipeline.py](corrected_pipeline.py).

---

## 4. Valoarea reală a performanței și flagul

- **Performanță reală (`LeaveOneGroupOut` pe 1,971 rânduri):** `1511 / 1971 = 0.7666` (**76.66%**)
- **Performanță reală (`LeaveOneOut` pe 900 aplicanți deduplicați):** `689 / 900 = 0.7656` (**76.56%**)
- **Flag:** **`SIGMOID_76`** (prin trunchiere la procent întreg `int(acc * 100)`, conform exemplului din enunț `0.8649 → SIGMOID_86`; dacă evaluatorul folosește rotunjire la cel mai apropiat întreg, valoarea corespunzătoare este **`SIGMOID_77`**).
