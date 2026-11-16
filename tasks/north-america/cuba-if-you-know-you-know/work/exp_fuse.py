# usage: exp_fuse.py <dense_tag> [<dense_tag2>]  -> tunes linear fusion of BM25 doc + dense doc scores
import sys, os, numpy as np, bm25s, Stemmer, collections
from common import *; from qnorm import *
ids, texts, meta = load_corpus(); Q = load_queries()
golds = np.array([ids.index(g) for _, _, g in Q])
cnt = collections.Counter(); [cnt.update(norm(t)) for t in texts]; sp = Speller(cnt)
st = Stemmer.Stemmer("english")
qs = [clean(q, sp) for _, q, _ in Q]
def bm25_scores(k1=1.5, b=0.9, corpus=None):
    r = bm25s.BM25(k1=k1, b=b); r.index(bm25s.tokenize(corpus, stopwords="en", stemmer=st, show_progress=False), show_progress=False)
    res, sc = r.retrieve(bm25s.tokenize(qs, stopwords="en", stemmer=st, show_progress=False), k=len(corpus), show_progress=False)
    S = np.zeros((len(qs), len(corpus)), np.float32)
    for i in range(len(qs)): S[i, res[i]] = sc[i]
    return S
B = bm25_scores(corpus=[" ".join(norm(t)) for t in texts])
def nz(S):  # per-query: divide by max
    return S / (S.max(1, keepdims=True) + 1e-9)
def zs(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
def h3(S): 
    top = np.argsort(-S, 1)[:, :3]; return float(np.mean([g in t for g, t in zip(golds, top)]))
Ds = [np.load(os.path.join("..", "runs", "dense", t, "docscores.npy")) for t in sys.argv[1:]]
print("bm25", h3(B), *[f"dense{i} {h3(D):.3f}" for i, D in enumerate(Ds)])
Bz = zs(B)
for D in Ds:
    Dz = zs(D)
    print(" ".join(f"w{w:.1f}={h3(Bz + w * Dz):.3f}" for w in [0.3,0.5,0.7,1.0,1.3,1.6,2.0,3.0]))
# 2-fold CV on weight to estimate honest number
rng = np.random.default_rng(0); idx = rng.permutation(len(qs)); folds = [idx[::2], idx[1::2]]
for D in Ds:
    Dz = zs(D); hits = []
    for f in range(2):
        tr, te = folds[f], folds[1 - f]
        def h(S, rows): top = np.argsort(-S[rows], 1)[:, :3]; return np.mean([g in t for g, t in zip(golds[rows], top)])
        ws = [0.3,0.5,0.7,1.0,1.3,1.6,2.0,3.0]
        w = max(ws, key=lambda w: h(Bz + w * Dz, tr)); hits.append(h(Bz + w * Dz, te)); 
    print("cv hit3", np.mean(hits))
