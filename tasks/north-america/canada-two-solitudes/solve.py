"""Two Solitudes (Canada) — soluția completă, reproductibilă.

    python3 -I solve.py        # ~3-4 minute; scrie model.npz și rulează open_vault.py pe el

Pași (detalii în README.md):
 1. m0b  — LR pe corpusul curățat: fără antet, fără propoziții generice (devin exemple neutre, cu toate
           cele 9 etichete), fără distractori; fiecare propoziție de hazard etichetată cu lexiconul nostru;
           ponderi echilibrate pe (clasă, limbă); celulele foarte frecvente (cifre, cuvinte de legătură) mascate.
 2. ml1  — același set, regresie logistică L1 (model rar, mai puțin zgomot din coliziunile hash).
 3. mn10 — augmentare care imită rapoartele reale (mai lungi, ~jumătate cuvinte nevăzute -> celule aleatoare).
 4. Ansamblu -> propagare pe graful kNN (idf) al amprentelor sigilate (fără etichete) -> etichete netezite.
 5. W liniar (ridge în spațiul nucleu) care reproduce scorurile pe rapoartele sigilate -> model.npz.
"""
import os, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
sys.path.insert(0, "files"); sys.path.insert(0, ".")
from open_vault import load_sealed

if not os.path.exists("Xs.npy"):
    np.save("Xs.npy", load_sealed())

import train, train_noise
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from featurize import featurize

# 1. modelul de bază pe corpusul curățat
train.main(rounds=0, out="m0b.npz")

# 2. modelul rar L1
texts, y, langs, gen = train.build()
X = featurize(texts); w = train.balance_weights(y, langs)
Xg = featurize([g[0] for g in gen]); cg = np.sqrt([g[2] for g in gen]); cg = 0.5 * cg * len(y) / cg.sum()
XA = np.vstack([X] + [Xg] * 9); yA = np.concatenate([y, np.repeat(np.arange(9), len(gen))])
wA = np.concatenate([w, np.tile(cg / 9, 9)])
Xs = np.load("Xs.npy")
mask = np.flatnonzero(((Xs > 0).mean(0) > 0.12) | ((X > 0).mean(0) > 0.12)); XA[:, mask] = 0
m = LogisticRegression(C=1.0, l1_ratio=1.0, solver="saga", max_iter=2000, tol=1e-3).fit(sparse.csr_matrix(XA), yA, sample_weight=wA)
W = m.coef_.copy(); W[:, mask] = 0; np.savez("ml1_1.0.npz", W=W, b=np.zeros(9))

# 3. modelul cu augmentare de zgomot
train_noise.main(C=10.0, noise_mean=60, out="mn10.npz")

# 4-5. ansamblu + propagare transductivă + W liniar final
subprocess.run([sys.executable, "-I", "transduce.py", "m0b.npz,ml1_1.0.npz,mn10.npz", "0.8", "model.npz"], check=True)

# verificare cu scriptul oficial
subprocess.run([sys.executable, "-I", "open_vault.py", os.path.join(HERE, "model.npz")], cwd="files", check=True)
