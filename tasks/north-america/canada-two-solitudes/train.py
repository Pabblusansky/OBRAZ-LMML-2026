# Model robust la schimbarea de domeniu (corpus sintetic -> rapoarte reale), EN + FR.
# 1) Curățăm corpusul: scoatem antetul (loc, dată) și propozițiile generice (fără cuvânt de hazard),
#    care poartă corelații false cu eticheta, diferite între engleză și franceză („două singurătăți").
# 2) Scoatem și propozițiile „distractor" (alt hazard decât eticheta raportului).
# 3) Echilibrăm ponderile pe (clasă, limbă), ca limba să nu mai prezică clasa.
# 4) Auto-antrenare pe amprentele sigilate (fără etichete): adăugăm rapoartele reale de care modelul e
#    cel mai sigur, cu eticheta prezisă, și reantrenăm — modelul învață vocabularul rapoartelor reale.
import csv, re, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, "files"); sys.path.insert(0, ".")
from featurize import featurize, tokens
from open_vault import CLASSES, load_sealed, try_vault
from lexicon import sentences, label_sentence

def lang(t):
    return "fr" if re.search(r"\b(les|des|est|sont|une|du|le|la)\b", t) else "en"

def build():
    rows = list(csv.DictReader(open("files/corpus.csv", encoding="utf-8")))
    texts, ys, langs = [], [], []
    generic = {}                                       # propoziții generice + antete: neutre față de clasă
    for r in rows:
        lab = CLASSES.index(r["label"]); L = lang(r["text"])
        keep = []
        ss = sentences(r["text"])
        generic.setdefault(" ".join(tokens(ss[0])), [ss[0], L, 0])[2] += 1
        for s in ss[1:]:                               # fără antet (loc, dată)
            ls = label_sentence(s)
            if ls is None:                             # propoziție generică -> exemplu neutru
                generic.setdefault(" ".join(tokens(s)), [s, L, 0])[2] += 1
                continue
            texts.append(s); ys.append(CLASSES.index(ls)); langs.append(L)   # instanță la nivel de propoziție
            if ls == r["label"]:
                keep.append(s)
        if keep:                                       # raportul curățat, doar propozițiile hazardului principal
            texts.append(" ".join(keep)); ys.append(lab); langs.append(L)
    return texts, np.array(ys), np.array(langs), list(generic.values())

def balance_weights(y, langs):
    w = np.ones(len(y))
    for c in range(len(CLASSES)):
        for L in ("en", "fr"):
            m = (y == c) & (langs == L)
            if m.any():
                w[m] = 1.0 / m.sum()
    return w * len(y) / w.sum()

MASK = None   # celule interzise (cifre, cuvinte de legătură): greutate forțată la 0

def fit(X, y, w, C):
    from scipy import sparse
    X = X.copy()
    if MASK is not None:
        X[:, MASK] = 0.0
    m = LogisticRegression(C=C, max_iter=3000).fit(sparse.csr_matrix(X), y, sample_weight=w)
    W = np.zeros((len(CLASSES), X.shape[1])); b = np.zeros(len(CLASSES))
    W[m.classes_], b[m.classes_] = m.coef_, m.intercept_
    return W, b

def main(C=10.0, rounds=6, frac0=0.3, frac1=0.8, gen_w=0.5, st_w=0.5, out="model.npz", mask_df=0.12):
    texts, y, langs, gen = build()
    X = featurize(texts)
    w = balance_weights(y, langs)
    # exemplele neutre: fiecare propoziție generică apare cu toate cele 9 etichete, pondere egală,
    # ca greutățile pe cifre, cuvinte de legătură, „au fost uciși" etc. să nu favorizeze nicio clasă
    Xg = featurize([g[0] for g in gen]); cg = np.array([g[2] for g in gen], float)
    cg = np.sqrt(cg); cg *= gen_w * len(y) / cg.sum()
    XG = np.vstack([Xg] * 9); yG = np.repeat(np.arange(9), len(gen)); wG = np.tile(cg / 9, 9)
    Xs = load_sealed()
    global MASK
    # celulele foarte frecvente (în >mask_df din rapoarte, reale sau sintetice) nu spun nimic despre hazard
    MASK = np.flatnonzero(((Xs > 0).mean(0) > mask_df) | ((X > 0).mean(0) > mask_df))
    print("masked buckets:", len(MASK))
    def train_with(extraX=None, extray=None, extraw=None):
        parts = [(X, y, w), (XG, yG, wG)] + ([(extraX, extray, extraw)] if extraX is not None else [])
        W, b = fit(np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                   np.concatenate([p[2] for p in parts]), C)
        b[:] = 0.0                                     # fără prior de clasă: distribuția reală e necunoscută
        W[:, MASK] = 0.0
        return W, b
    W, b = train_with()
    S = Xs @ W.T + b
    print("round 0 preds:", np.bincount(S.argmax(1), minlength=9), "vault:", try_vault(S, verbose=False))
    hist = [(W, b)]
    for r in range(1, rounds + 1):
        frac = frac0 + (frac1 - frac0) * (r - 1) / max(1, rounds - 1)
        part = np.sort(S, 1); marg = part[:, -1] - part[:, -2]
        pred = S.argmax(1)
        sel, ws = [], []
        for c in range(len(CLASSES)):                  # cei mai siguri, pe fiecare clasă, pondere egală pe clasă
            idx = np.flatnonzero(pred == c)
            idx = idx[np.argsort(-marg[idx])][: max(1, int(len(idx) * frac))]
            sel.extend(idx); ws.extend([1.0 / len(idx)] * len(idx))
        sel = np.array(sel); ws = np.array(ws); ws *= st_w * len(y) / ws.sum()
        W, b = train_with(Xs[sel], pred[sel], ws)
        S = Xs @ W.T + b
        print(f"round {r} (frac {frac:.2f}) preds:", np.bincount(S.argmax(1), minlength=9),
              "changed:", int((S.argmax(1) != pred).sum()), "vault:", try_vault(S, verbose=False))
        hist.append((W, b))
    np.savez(out, W=W, b=b)
    return hist

if __name__ == "__main__":
    main()
