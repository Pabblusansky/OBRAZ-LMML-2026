# usage: exp_all.py name=path.npy ...   fuse z-scored signals; weights by coordinate ascent, honest number by repeated 2-fold CV
import sys, numpy as np, itertools
from common import *
ids, _, _ = load_corpus(); Q = load_queries()
golds = np.array([ids.index(g) for _, _, g in Q])
zs = lambda S: (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
names, Z = [], []
for a in sys.argv[1:]:
    n, p = a.split("="); names.append(n); Z.append(zs(np.load(p).astype(np.float32)))
Z = np.stack(Z)  # signals x queries x docs
gold_s = Z[:, np.arange(len(golds)), golds]  # signals x queries
def hits(w, rows):
    S = np.tensordot(w, Z[:, rows], 1); g = (w @ gold_s[:, rows])[:, None]
    return float(np.mean((S > g).sum(1) < 3))
GRID = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]
def fit(rows):
    w = np.zeros(len(names)); w[0] = 1.0
    for _ in range(3):
        for j in range(1, len(names)):
            best = max(GRID, key=lambda v: (hits(np.r_[w[:j], v, w[j+1:]], rows), -abs(v - 0.5)))
            w[j] = best
    return w
allr = np.arange(len(golds))
w = fit(allr); print("weights", dict(zip(names, w)), "fit-all hit3", hits(w, allr))
for j, n in enumerate(names): print(f"  alone {n}: {hits(np.eye(len(names))[j], allr):.3f}")
cv = []
for seed in range(5):
    p = np.random.default_rng(seed).permutation(len(golds)); a, b = p[::2], p[1::2]
    cv += [hits(fit(a), b), hits(fit(b), a)]
print(f"CV hit3 mean={np.mean(cv):.3f} min={np.min(cv):.3f}")
