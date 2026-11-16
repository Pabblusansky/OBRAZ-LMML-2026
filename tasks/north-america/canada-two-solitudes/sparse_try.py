# Model rar (L1): puține celule cu greutate -> mai puțin zgomot din coliziunile hash ale cuvintelor nevăzute
import sys, numpy as np
sys.path.insert(0, "files"); sys.path.insert(0, ".")
from scipy import sparse
from sklearn.linear_model import LogisticRegression
import train
from featurize import featurize
from open_vault import try_vault
texts, y, langs, gen = train.build()
X = featurize(texts); w = train.balance_weights(y, langs)
Xg = featurize([g[0] for g in gen]); cg = np.sqrt([g[2] for g in gen]); cg = 0.5 * cg * len(y) / cg.sum()
XA = np.vstack([X] + [Xg] * 9); yA = np.concatenate([y, np.repeat(np.arange(9), len(gen))]); wA = np.concatenate([w, np.tile(cg / 9, 9)])
Xs = np.load("Xs.npy")
mask = np.flatnonzero(((Xs > 0).mean(0) > 0.12) | ((X > 0).mean(0) > 0.12)); XA[:, mask] = 0
f = np.load("files/first_model.npz"); pf = (Xs @ f["W"].T + f["b"]).argmax(1)
for C in [float(c) for c in sys.argv[1:]]:
    m = LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=2000, tol=1e-3).fit(sparse.csr_matrix(XA), yA, sample_weight=wA)
    W = m.coef_.copy(); W[:, mask] = 0
    S = Xs @ W.T
    print(C, "nnz buckets", int((np.abs(W).sum(0) > 0).sum()), np.bincount(S.argmax(1), minlength=9),
          "ties0", int((S.max(1) == 0).sum()), "agree first", round((S.argmax(1) == pf).mean(), 3), try_vault(S, verbose=False))
    np.savez(f"ml1_{C}.npz", W=W, b=np.zeros(9))
