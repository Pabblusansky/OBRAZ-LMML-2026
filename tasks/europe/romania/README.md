# România – Shaken, Still Standing → `SIGMOID_CONVOLUTION`

Rulare: `python3 -I solution.py` (log complet în `solution_output.txt`, predicții în `predictions.csv`).

## Evaluatorul
`evaluation_script.py` decodează un cod Reed–Solomon (p = 65521, n = 5000, k = 4252): fiecare predicție alege o valoare din `vault.table`; flag-ul apare doar dacă ≥ (n+k)/2 = 4626 predicții sunt corecte, adică **acuratețe ≥ 92,52%**. Cuvântul e verificat cu un tag SHA-256, deci nu poate fi un fals pozitiv.

## Ce am descoperit în date
1. **12 canale curate** (c03, c09, c12, c14, c15, c19, c27, c28, c30, c36, c39, c40) — gaussiene, kurtosis ≈ 0. Celelalte 36 au zgomot cu amplitudine dependentă de rețea (kurtosis ≈ 0,9 = amestec de scale).
2. **Transilvania** = grupul cu zgomot de ~4× mai mic: un GMM cu 2 componente îl separă exact (1.900 train / 2.000 sealed). Eticheta depinde **pătratic** de cele 12 canale curate (clasele diferă prin covarianță).
3. În grupul mare, **c13 este bimodal** (≈ −1,8 vs ≈ −0,1) și indică rețeaua:
   - **Moldova** (c13 mic): eticheta = paritate/XOR a pragurilor `c35 > −0,18`, `c43 > −0,61`, `c38 > 0,72` (găsită cu un arbore de adâncime 4; 94,6% doar cu regula);
   - **Muntenia** (c13 mare): eticheta e **liniară** (regresie logistică 93,1% pe subgrup; ponderi mari pe c12, c30, c01, c47, c41).

## Modele încercate (CV 5-fold, acuratețe)
| Model | Date | Scor |
|---|---|---|
| Logistic regression (baseline) | toate | 0,601 |
| SVM rbf (baseline) | toate | 0,675 |
| Random forest | toate | 0,764 |
| kNN | toate | 0,528 |
| HistGradientBoosting | toate | 0,853 |
| Logistic (liniar) | Transilvania, 12 canale | 0,583 |
| QDA | Transilvania | 0,898 |
| SVM rbf C=3 | Transilvania | 0,915 |
| MLP | Transilvania | 0,866 |
| HGB | Transilvania | 0,833 |
| **Poly-2 logistic (C=0,1)** | Transilvania | **0,930** |
| Logistic | Moldova+Muntenia | 0,628 |
| HGB brut | Moldova+Muntenia | 0,916 |
| Mixture-of-experts EM (gate GMM pe c13) | Moldova+Muntenia | 0,883 |
| **HGB + bit paritate + scor expert liniar (stacking)** | Moldova+Muntenia | **0,947** |

## Combinarea
- Rutare pe rețea: GMM nesupervizat (train+sealed) → Transilvania vs. restul.
- Transilvania → logistică polinomială de grad 2.
- Moldova+Muntenia → stacking: HGB primește canalele brute + bitul de paritate (expertul Moldova) + scorul out-of-fold al logisticii antrenate pe rândurile cu c13 mare (expertul Muntenia); media pe 5 seed-uri.
- Acuratețe estimată pe sealed: 0,6·0,947 + 0,4·0,930 ≈ **0,940** > 0,925 → evaluatorul afișează `SIGMOID_CONVOLUTION`.
