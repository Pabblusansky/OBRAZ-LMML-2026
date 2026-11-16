"""Shaken, Still Standing (România) – soluție.

Algoritmul, pas cu pas:

1. Cele 48 de canale nu sunt toate la fel. 12 canale (c03, c09, c12, c14, c15, c19, c27,
   c28, c30, c36, c39, c40) au distribuție gaussiană curată; celelalte 36 sunt „zgomot” a
   cărui amplitudine depinde de rețeaua care a trimis înregistrarea.
2. Separăm rețelele:
   - un GMM cu 2 componente pe toate canalele găsește grupul cu zgomot mic
     (exact 1.900 în train / 2.000 în sealed) -> rețeaua „Transilvania”;
   - în grupul mare, canalul c13 este bimodal: c13 mic -> „Moldova”, c13 mare -> „Muntenia”.
3. Fiecare rețea are altă regulă de etichetare, deci alt model potrivit:
   - Transilvania: regula e pătratică (clasa 1 are altă covarianță) -> regresie logistică
     pe trăsături polinomiale de grad 2 din cele 12 canale curate;
   - Moldova: regulă de tip paritate/XOR pe praguri la c35, c43, c38 (găsită cu un arbore
     de decizie de adâncime 4) -> un model liniar nu o poate învăța, arborii da;
   - Muntenia: regulă liniară -> regresie logistică.
4. Combinarea (stacking) pentru grupul mare: HistGradientBoosting primește canalele brute
   + bitul de paritate + scorul expertului liniar (antrenat doar pe rândurile cu c13 mare,
   calculat out-of-fold pe train ca să nu „vadă” eticheta). Boosting-ul învață singur când
   să folosească paritatea și când scorul liniar, în funcție de c13.
5. Scriem predictions.csv și rulăm evaluatorul (Reed–Solomon: cere ~92,5% acuratețe).
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

HERE = Path(__file__).resolve().parent
C = [f"c{j + 1:02d}" for j in range(48)]
CLEAN = ["c03", "c09", "c12", "c14", "c15", "c19", "c27", "c28", "c30", "c36", "c39", "c40"]
LIN = [c for c in C if c != "c13"]  # c13 e doar „indicatorul de rețea”, nu intră în expertul liniar
CV = StratifiedKFold(5, shuffle=True, random_state=0)

train = pd.read_csv(HERE / "dataset" / "train.csv")
sealed = pd.read_csv(HERE / "dataset" / "sealed.csv")
nt = len(train)

# --- Pasul 2: identificăm rețeaua Transilvania (zgomot mic) cu un GMM nesupervizat ------
# Folosim și train și sealed: e nesupervizat, deci nu se scurge nicio etichetă.
allX = np.vstack([train[C].values, sealed[C].values])
gmm = GaussianMixture(2, covariance_type="full", n_init=3, random_state=0).fit(allX)
lab = gmm.predict(allX)
small_id = np.argmin(np.bincount(lab))  # componenta mică = Transilvania
is_tr = lab == small_id
tr_train, tr_sealed = is_tr[:nt], is_tr[nt:]
print("Transilvania: train", tr_train.sum(), "sealed", tr_sealed.sum())


# --- Pasul 3a: expertul pătratic pentru Transilvania ---------------------------------------
def quad_model():
    # grad 2 => termenii x_i^2 și x_i*x_j surprind diferența de covarianță între clase
    return make_pipeline(StandardScaler(), PolynomialFeatures(2), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=10000))


T = train[tr_train]
scores = {}
scores["Transilvania: logistic (liniar)"] = cross_val_score(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), T[CLEAN], T.vrancea, cv=CV).mean()
scores["Transilvania: HGB"] = cross_val_score(HistGradientBoostingClassifier(), T[CLEAN], T.vrancea, cv=CV).mean()
scores["Transilvania: poly2 logistic"] = cross_val_score(quad_model(), T[CLEAN], T.vrancea, cv=CV).mean()


# --- Pasul 3b: bitul de paritate (Moldova) -------------------------------------------------
def parity(D):
    # pragurile vin din arborele de decizie de adâncime 4 antrenat pe grupul mare
    return ((D.c35 > -0.18).astype(int) ^ (D.c43 > -0.61).astype(int) ^ (D.c38 > 0.72).astype(int)).values


def lin_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))


B = train[~tr_train].reset_index(drop=True)
S = sealed[~tr_sealed].reset_index(drop=True)
yB = B.vrancea.values

# --- Pasul 3c: scorul expertului liniar (Muntenia), out-of-fold pe train --------------------
# Expertul e antrenat doar pe rânduri cu c13 mare (majoritar Muntenia), ca să învețe regula liniară
# fără să fie „stricat” de rândurile Moldova, unde eticheta e paritate.
lin_oof = np.zeros(len(B))
for a, b in CV.split(B, yB):
    ha = a[B.c13.values[a] > -0.3]
    lin_oof[b] = lin_model().fit(B.loc[ha, LIN], yB[ha]).decision_function(B.loc[b, LIN])
hi = B.c13.values > -0.3
lin_full = lin_model().fit(B.loc[hi, LIN], yB[hi])
lin_sealed = lin_full.decision_function(S[LIN])

# --- Pasul 4: stacking – HGB pe canale brute + paritate + scor liniar ----------------------
FB = np.c_[B[C].values, parity(B), lin_oof]
FS = np.c_[S[C].values, parity(S), lin_sealed]
CV2 = StratifiedKFold(5, shuffle=True, random_state=1)
scores["Moldova+Muntenia: logistic"] = cross_val_score(lin_model(), B[C], yB, cv=CV2).mean()
scores["Moldova+Muntenia: HGB brut"] = cross_val_score(
    HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400), B[C], yB, cv=CV2).mean()
scores["Moldova+Muntenia: HGB + paritate + scor liniar"] = cross_val_score(
    HistGradientBoostingClassifier(learning_rate=0.03, max_iter=600), FB, yB, cv=CV2).mean()

print("\nAcuratețe cross-validation (5-fold):")
for k, v in scores.items():
    print(f"  {k:50s} {v:.4f}")

# media mai multor seed-uri de boosting reduce varianța predicției
pB = np.mean([HistGradientBoostingClassifier(learning_rate=0.03, max_iter=600, random_state=r)
              .fit(FB, yB).predict_proba(FS)[:, 1] for r in range(5)], axis=0)
pT = quad_model().fit(T[CLEAN], T.vrancea).predict_proba(sealed[tr_sealed][CLEAN])[:, 1]

pred = np.zeros(len(sealed), dtype=int)
pred[~tr_sealed] = (pB > 0.5).astype(int)
pred[tr_sealed] = (pT > 0.5).astype(int)
w_big, w_sm = (~tr_sealed).mean(), tr_sealed.mean()
est = w_big * scores["Moldova+Muntenia: HGB + paritate + scor liniar"] + w_sm * scores["Transilvania: poly2 logistic"]
print(f"\nacuratețe estimată pe sealed: {est:.4f} (prag necesar 0.9252)")

out = pd.DataFrame({"record_id": sealed.record_id, "vrancea": pred})
out.to_csv(HERE / "predictions.csv", index=False)
print("predictions.csv:", out.vrancea.value_counts().to_dict())

# --- Pasul 5: evaluatorul ------------------------------------------------------------------
sys.stdout.flush()
subprocess.run([sys.executable, "-I", str(HERE / "evaluation_script.py"), str(HERE / "predictions.csv")], check=False)
