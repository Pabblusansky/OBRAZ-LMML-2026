import numpy as np, sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
d = np.load(sys.argv[1], allow_pickle=True)
E, F, T, y = d["image_embeddings"], d["flipped_embeddings"], d["text_embeddings"], d["labels"]
brands = ["nike","adidas","converse","puma","reebok","vans","asics"]
REJECT_KNOWN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
def scores(trE, trF, try_, teE, known_idx):
    clf = LogisticRegression(C=10, max_iter=3000).fit(np.vstack([trE, trF]), np.r_[try_, try_])
    P = clf.predict_proba(teE); pred = clf.classes_[P.argmax(1)]
    S = teE @ trE.T
    knn = np.array([np.sort(S[i][try_ == pred[i]])[-5:].mean() for i in range(len(teE))])
    Z = np.exp(100 * teE @ T.T); Z /= Z.sum(1, keepdims=True)
    unk_cols = [i for i in range(7) if i not in known_idx]
    zs = Z[:, unk_cols].sum(1)          # prob mass on brands we were NOT trained on
    return pred, P.max(1), knn, zs
for held in ["nike","adidas","converse"]:
    known = [b for b in ["nike","adidas","converse"] if b != held]; kidx = [brands.index(b) for b in known]
    km = np.isin(y, known)
    # CV scores on knowns (to set thresholds) + scores on held-out brand (unknown)
    kn = {k: np.zeros(km.sum()) for k in ["prob","knn","zs"]}; kpred = np.empty(km.sum(), object)
    Ek, Fk, yk = E[km], F[km], y[km]
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Ek, yk):
        p, a, b, c = scores(Ek[tr], Fk[tr], yk[tr], Ek[te], kidx); kpred[te] = p; kn["prob"][te], kn["knn"][te], kn["zs"][te] = a, b, c
    _, ua, ub, uc = scores(Ek, Fk, yk, E[~km], kidx); un = {"prob": ua, "knn": ub, "zs": -uc}
    kn["zs"] = -kn["zs"]   # all scores: higher = more 'known'
    # combined: average of ranks relative to known distribution
    def pct(v, ref): return np.searchsorted(np.sort(ref), v) / len(ref)
    kn["combo"] = np.mean([pct(kn[k], kn[k]) for k in ["prob","knn","zs"]], 0)
    un["combo"] = np.mean([pct(un[k], kn[k]) for k in ["prob","knn","zs"]], 0)
    kn["knn+zs"] = np.mean([pct(kn[k], kn[k]) for k in ["knn","zs"]], 0)
    un["knn+zs"] = np.mean([pct(un[k], kn[k]) for k in ["knn","zs"]], 0)
    line = []
    for k in ["prob","knn","zs","combo","knn+zs"]:
        thr = np.quantile(kn[k], REJECT_KNOWN)
        known_acc = ((kpred == yk) & (kn[k] >= thr)).mean(); rej = (un[k] < thr).mean()
        line.append(f"{k}: acc {known_acc:.2f} rej {rej:.2f}")
    print(f"unknown={held:9s} | " + " | ".join(line))
