# Picks the fusion weights on the public questions using the real Retriever, reports a cross-validated hit@3,
# and writes them to submission/index/config.json.   usage: python tune.py [--dry]
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "..", "submission")
sys.path.insert(0, SUB)
from main import Retriever, bm25_terms, clean_query, zscore
from common import load_queries

r = Retriever(os.path.join(SUB, "index"))
Q = load_queries()
golds = np.array([r.doc_ids.index(g) for _, _, g in Q])
names = ["doc", "title", "chunk", "dense"]
sig = {n: [] for n in names}
for _, q, _ in Q:
    terms = bm25_terms(clean_query(q, r.speller))
    sig["doc"].append(zscore(r.bm25_doc.scores(terms)))
    sig["title"].append(zscore(r.bm25_title.scores(terms)))
    sig["chunk"].append(zscore(np.maximum.reduceat(r.bm25_chunk.scores(terms), r.bm25_chunk_starts)))
    sig["dense"].append(zscore(np.maximum.reduceat(r.chunk_vecs @ r.embed(q), r.dense_starts)))
Z = np.stack([np.stack(sig[n]) for n in names])
G = Z[:, np.arange(len(golds)), golds]


def hits(w, rows):
    S = np.tensordot(w, Z[:, rows], 1); g = (w @ G[:, rows])[:, None]
    return float(np.mean((S > g).sum(1) < 3))


GRID = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]


def fit(rows):
    w = np.array([1.0, 0.3, 0.7, 1.0])
    for _ in range(3):
        for j in range(1, len(names)):
            w[j] = max(GRID, key=lambda v: (hits(np.r_[w[:j], v, w[j + 1:]], rows), -abs(v - w[j])))
    return w


allr = np.arange(len(golds))
for j, n in enumerate(names):
    print(f"alone {n:6s} hit3={hits(np.eye(4)[j], allr):.3f}")
w = fit(allr)
print("weights", dict(zip(names, w.round(2))), f"fit-all hit3={hits(w, allr):.3f}")
cv = []
for seed in range(10):
    p = np.random.default_rng(seed).permutation(len(golds)); a, b = p[::2], p[1::2]
    cv += [hits(fit(a), b), hits(fit(b), a)]
print(f"2-fold CV x10 hit3: mean={np.mean(cv):.3f} min={np.min(cv):.3f} max={np.max(cv):.3f}")
if "--dry" not in sys.argv:
    json.dump({"weights": dict(zip(names, map(float, w)))}, open(os.path.join(SUB, "index", "config.json"), "w"), indent=1)
    print("written config.json")
