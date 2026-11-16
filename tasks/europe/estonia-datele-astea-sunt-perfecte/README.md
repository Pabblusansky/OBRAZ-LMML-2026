# Estonia – Datele astea sunt perfecte!

| Câmp | Valoare |
|---|---|
| Țara | Estonia |
| Dificultate | Mediu |
| Categorie | audit ML / scurgere de date (data leakage) |
| Rezolvat de | Dragos |
| Status | rezolvat |
| Flag | `SIGMOID_76` (trunchiere, ca în exemplul `0.8649 → SIGMOID_86`); `SIGMOID_77` dacă se rotunjește |

## Problema pe scurt

Ministerul estonian raportează o acuratețe pe test de **93,42%** (369/395) pentru
`files/project_pipeline.py`: `StandardScaler` + `KNeighborsClassifier(n_neighbors=1)` pe
6 caracteristici numerice ale celor 1971 de cereri e-Residency (`label` binar = fraudă
sau nu), cu `train_test_split(test_size=0.2, random_state=1137)`.

Trebuie: (1) găsită alegerea care umflă artificial scorul, (2) demonstrat experimental
că celelalte suspiciuni sunt piste false, (3) reparat bug-ul păstrând același model și
(4) evaluată acuratețea reală cu `LeaveOneOut` (sau echivalentul pe grupuri) +
`cross_val_predict`. Flag-ul derivă din acuratețea reală.

## Ideea: de ce 1-NN „trișează” când datele au duplicate

1-NN (un singur vecin) clasifică un punct copiind eticheta celui mai apropiat punct din
antrenare. Dacă aceeași persoană apare de mai multe ori cu date aproape identice, iar o
cerere ajunge în train și alta în test, vecinul cel mai apropiat al cererii de test e
**propria clonă** → răspuns corect garantat, fără nicio generalizare.

## Bug-ul adevărat: scurgere la nivel de aplicant

Pipeline-ul aruncă coloanele de identitate (`submission_id`, `date_of_birth`,
`citizenship`) și apoi împarte aleatoriu pe **rânduri**. Dar:

- cele 1971 de rânduri provin de la doar **900 de persoane** (perechea
  `(date_of_birth, citizenship)`): 360 cu 1 cerere, 195 cu 2, 159 cu 3, 186 cu 4;
- toate cererile unei persoane au **aceeași etichetă** și caracteristici aproape
  identice (abaterea intra-grup ≈ 1% din cea totală);
- după scalare, distanța **maximă** dintre cererile aceleiași persoane (0,1018) e mai
  mică decât distanța **minimă** dintre persoane diferite (0,2812).

Pe split-ul original: 308 din 395 de rânduri de test (78%) au o clonă în train →
acuratețe 100%; restul de 87 (persoane noi) → ~70%. Media ponderată
`(308·1,0 + 87·0,70) / 395 ≈ 0,934` explică exact scorul raportat.

## Pistele false (testate și infirmate)

| Ipoteză | Reparația testată | Acuratețe (rs=1137) | Alte verificări |
|---|---|---|---|
| Original | – | **0,9342** | LOO naiv: 0,9594 |
| A: scurgere în `StandardScaler` (fit pe tot setul) | scaler în `make_pipeline`, fit doar pe train | 0,9367 | 100 de seed-uri: 0,9440 ± 0,0113 (vs 0,9443 cu scurgere) |
| B1: 41 de valori imposibile (negative, `doc_scan_quality > 1`) – eliminare | se elimină rândurile | 0,9430 | LOO: 0,9611 |
| B2: aceleași valori – plafonare (clip) | `≥ 0`, `[0, 1]` | 0,9316–0,9342 | LOO: 0,9589 |
| C: split nestratificat / un singur seed | `stratify=y`, `StratifiedKFold(5)`, LOO naiv | 0,94 | 0,9432 / 0,9594 |

Toate rămân în banda 93–96%. LOO naiv chiar **crește** scorul, pentru că fiecare rând de
test are până la 3 clone în train.

## Reparația și acuratețea reală

Păstrăm modelul cerut, dar validăm **pe grupuri de aplicanți**, ca o persoană să nu fie
niciodată și în train și în test:

```python
groups = df.groupby(["date_of_birth", "citizenship"]).ngroup()
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
preds = cross_val_predict(pipe, X, y, cv=LeaveOneGroupOut(), groups=groups)
accuracy_score(y, preds)   # 0.7666 (1511 / 1971)
```

Verificare echivalentă: deduplicare la o cerere per persoană (900 rânduri) +
`LeaveOneOut()` → 0,7656 (689/900). Ambele dau ~76,6% → `SIGMOID_76`.

## Cum funcționează scripturile

### `main.py` (auditul complet + generează `solution.txt`)
1. `load_dataset` – găsește și citește `applications.csv`.
2. `make_model_pipeline` – `StandardScaler` + 1-NN într-un `Pipeline` (scalerul se
   antrenează doar pe datele de antrenare din fiecare fold).
3. `evaluate_original_pipeline` – reproduce exact codul original (0,9342).
4. `audit_false_lead_preprocessing` – pista A: scaler în pipeline, media pe 100 de seed-uri, LOO.
5. `find_impossible_mask` / `clip_impossible_values` / `audit_false_lead_impossible_values` – pista B.
6. `audit_false_lead_split_config` – pista C: stratificare, `StratifiedKFold`, LOO naiv.
7. `audit_applicant_leakage` – construiește grupurile, numără cererile per persoană,
   verifică etichetele, calculează distanțele intra/inter-grup (`pairwise_distances`) și
   desparte acuratețea pe rânduri „scurse” vs persoane noi.
8. `evaluate_corrected_pipeline` – `LeaveOneGroupOut` + varianta deduplicată cu LOO;
   calculează flag-ul trunchiat și rotunjit.
9. `format_solution_text` – scrie raportul în română în `solution.txt`.

## Rulare

```bash
python main.py
```

## Fișiere

| Fișier | Rol |
|---|---|
| `task.md` | enunțul |
| `main.py` | auditul complet, generează `solution.txt` |
| `solution.md` | rezolvarea redactată (pentru link pe platformă) |
| `solution.txt` | raportul generat de `main.py` |
| `files/applications.csv`, `files/project_pipeline.py` | datele și pipeline-ul original |
