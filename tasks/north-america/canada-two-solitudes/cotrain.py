# O rundă de auto-antrenare: etichetele propagate ale rapoartelor reale + corpusul curățat -> LR nou -> propagare.
import sys, os, numpy as np
sys.path.insert(0, "files"); sys.path.insert(0, ".")
from scipy import sparse
from sklearn.linear_model import LogisticRegression
import train, transduce as T
from featurize import featurize
from vault_info import info
texts, y, langs, gen = train.build()
X = featurize(texts); w = train.balance_weights(y, langs)
Xs = T.Xs; G = T.knn_graph()
P0 = T.base_probs(["m0b.npz", "ml1_1.0.npz", "mn10.npz"])
F = T.propagate(P0, G, 0.8)
for r in range(int(sys.argv[1]) if len(sys.argv) > 1 else 2):
    lab = F.argmax(1); conf = F.max(1)
    sel = np.argsort(-conf)[: int(0.7 * len(lab))]
    ws = np.full(len(sel), 0.5 * len(y) / len(sel))
    XA = np.vstack([X, Xs[sel]]); XA[:, T.mask] = 0
    m = LogisticRegression(C=3.0, max_iter=2000).fit(sparse.csr_matrix(XA), np.concatenate([y, lab[sel]]),
                                                    sample_weight=np.concatenate([w, ws]))
    Wm = m.coef_.copy(); Wm[:, T.mask] = 0
    np.savez("mct.npz", W=Wm, b=np.zeros(9))
    P0 = (T.base_probs(["mct.npz"]) + T.base_probs(["m0b.npz", "ml1_1.0.npz", "mn10.npz"])) / 2
    F = T.propagate(P0, G, 0.8)
    W, b = T.fit_linear(F)
    np.savez(f"mct_final{r}.npz", W=W, b=b)
    print(r, np.bincount(F.argmax(1), minlength=9), "changed vs prev labels", int((F.argmax(1) != lab).sum()),
          info(Xs @ W.T + b))
