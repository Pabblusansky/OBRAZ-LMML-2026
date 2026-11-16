# Antrenare cu augmentare care imită rapoartele reale:
#  - rapoarte mai lungi: textul hazardului + propoziții generice aleatoare (din ambele limbi) + antet;
#  - cuvinte nevăzute în corpus: ~jumătate din termenii unui raport real lipsesc din vocabularul corpusului
#    și cad, prin hash, în celule aleatoare -> adăugăm celule aleatoare la amprentă.
# Astfel regresia logistică învață să nu se bazeze pe celule care apar „din întâmplare".
import csv, re, sys, math
import numpy as np
sys.path.insert(0, "files"); sys.path.insert(0, ".")
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from featurize import featurize_one, tokens, bucket, DIM
from open_vault import CLASSES, load_sealed, try_vault
from lexicon import sentences, label_sentence
import train

rng = np.random.default_rng(0)

def fp_from_tokens(toks, extra):
    """Amprenta ca în featurize, plus `extra` celule aleatoare (termeni nevăzuți)."""
    counts = {}
    for t in toks: b = bucket(t); counts[b] = counts.get(b, 0) + 1
    for a, b2 in zip(toks, toks[1:]): b = bucket(a + " " + b2); counts[b] = counts.get(b, 0) + 1
    for b in rng.integers(0, DIM, extra): counts[b] = counts.get(b, 0) + 1
    v = np.zeros(DIM, np.float32)
    for b, c in counts.items(): v[b] = 1.0 + math.log(c)
    return v / (np.linalg.norm(v) + 1e-12)

def build_aug(copies=3, noise_mean=60):
    rows = list(csv.DictReader(open("files/corpus.csv", encoding="utf-8")))
    gens = {"en": [], "fr": []}; items = []
    for r in rows:
        L = train.lang(r["text"]); ss = sentences(r["text"])
        hz = []
        for s in ss[1:]:
            ls = label_sentence(s)
            if ls is None: gens[L].append(s)
            elif ls == r["label"]: hz.append(s)
            else: hz.append(s) if rng.random() < 0.3 else None   # păstrăm uneori distractorii (cauze secundare)
        if any(label_sentence(s) == r["label"] for s in hz):
            items.append((ss[0], hz, CLASSES.index(r["label"]), L))
    X, y, langs = [], [], []
    for head, hz, lab, L in items:
        for c in range(copies):
            g = list(rng.choice(gens[L], rng.integers(0, 5)))
            parts = hz + g; rng.shuffle(parts)
            text = (head + " " if rng.random() < 0.5 else "") + " ".join(parts)
            X.append(fp_from_tokens(tokens(text), rng.poisson(noise_mean))); y.append(lab); langs.append(L)
    # exemple pur generice (fără hazard) cu toate etichetele -> neutre
    Xg = []
    for L in ("en", "fr"):
        for _ in range(1500):
            text = " ".join(rng.choice(gens[L], rng.integers(1, 5)))
            Xg.append(fp_from_tokens(tokens(text), rng.poisson(noise_mean)))
    return np.array(X), np.array(y), np.array(langs), np.array(Xg)

def main(C=1.0, noise_mean=60, copies=3, gen_w=0.3, out="mn.npz"):
    X, y, langs, Xg = build_aug(copies, noise_mean)
    w = train.balance_weights(y, langs)
    wg = np.full(len(Xg) * 9, gen_w * len(y) / (len(Xg) * 9))
    XA = sparse.csr_matrix(np.vstack([X] + [Xg] * 9)); yA = np.concatenate([y, np.repeat(np.arange(9), len(Xg))])
    m = LogisticRegression(C=C, max_iter=2000).fit(XA, yA, sample_weight=np.concatenate([w, wg]))
    W = m.coef_; b = np.zeros(9)
    Xs = load_sealed(); S = Xs @ W.T
    print(f"C={C} noise={noise_mean}", np.bincount(S.argmax(1), minlength=9), "vault:", try_vault(S, verbose=False))
    np.savez(out, W=W, b=b)

if __name__ == "__main__":
    main(C=float(sys.argv[1]), noise_mean=int(sys.argv[2]), out=sys.argv[3])
