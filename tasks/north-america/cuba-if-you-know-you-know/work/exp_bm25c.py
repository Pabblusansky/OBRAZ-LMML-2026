import numpy as np, bm25s, Stemmer, collections, os
from common import *; from qnorm import *
ids, texts, meta = load_corpus(); Q = load_queries()
golds = np.array([ids.index(g) for _, _, g in Q])
cnt = collections.Counter(); [cnt.update(norm(t)) for t in texts]; sp = Speller(cnt)
st = Stemmer.Stemmer("english")
qs = [clean(q, sp) for _, q, _ in Q]
qt = bm25s.tokenize(qs, stopwords="en", stemmer=st, show_progress=False)
def full(corpus, k1, b):
    r = bm25s.BM25(k1=k1, b=b); r.index(bm25s.tokenize(corpus, stopwords="en", stemmer=st, show_progress=False), show_progress=False)
    res, sc = r.retrieve(qt, k=len(corpus), show_progress=False)
    S = np.zeros((len(qs), len(corpus)), np.float32)
    for i in range(len(qs)): S[i, res[i]] = sc[i]
    return S
zs = lambda S: (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
def h3(S): top = np.argsort(-S, 1)[:, :3]; return float(np.mean([g in t for g, t in zip(golds, top)]))
def title(m):
    t = m["title"] if m["kind"] != "book_chapter" else m["title"]
    return t
B = full([" ".join(norm(t)) for t in texts], 1.5, 0.9); print("doc", h3(B))
for rep in [1, 3, 5]:
    Bt = full([(" " + title(m)) * rep + " " + " ".join(norm(t)) for m, t in zip(meta, texts)], 1.5, 0.9); print("title x", rep, h3(Bt))
T = full([title(m) + " " + m.get("source_title", "") for m in meta], 1.2, 0.75)
print("title-only field fused:", " ".join(f"{w}:{h3(zs(B) + w * zs(T)):.3f}" for w in [0.1,0.2,0.3,0.5]))
for size, stride in [(150, 75), (300, 150)]:
    chunks, cdoc = [], []
    for d, t in enumerate(texts):
        for c in chunk_words(" ".join(norm(t)), size, stride): chunks.append(c); cdoc.append(d)
    cdoc = np.array(cdoc); order = np.argsort(cdoc, kind="stable"); starts = np.searchsorted(cdoc[order], np.arange(len(ids)))
    C = np.maximum.reduceat(full(chunks, 1.2, 0.75)[:, order], starts, axis=1)
    np.save(f"../runs/bm25chunk_{size}.npy", C)
    print("chunk", size, h3(C), " fused:", " ".join(f"{w}:{h3(zs(B) + w * zs(C)):.3f}" for w in [0.3,0.5,0.7,1.0]))
np.save("../runs/bm25doc.npy", B)
