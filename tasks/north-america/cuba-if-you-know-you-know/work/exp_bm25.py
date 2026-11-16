import sys, numpy as np, bm25s, Stemmer
from common import *
ids, texts, meta = load_corpus(); Q = load_queries()
golds = [g for _, _, g in Q]; qtexts = [q for _, q, _ in Q]
st = Stemmer.Stemmer("english")
def run(size, stride, agg="max"):
    if size == 0:
        chunks, cdoc = texts, list(range(len(texts)))
    else:
        chunks, cdoc = [], []
        for d, t in enumerate(texts):
            for c in chunk_words(t, size, stride): chunks.append(c); cdoc.append(d)
    cdoc = np.array(cdoc)
    r = bm25s.BM25(k1=1.2, b=0.75)
    r.index(bm25s.tokenize(chunks, stopwords="en", stemmer=st, show_progress=False), show_progress=False)
    res, sc = r.retrieve(bm25s.tokenize(qtexts, stopwords="en", stemmer=st, show_progress=False), k=min(300, len(chunks)), show_progress=False)
    ranks = []
    for row in res:
        seen = []
        for c in row:
            d = ids[cdoc[c]]
            if d not in seen: seen.append(d)
        ranks.append(seen)
    print(f"size={size} stride={stride} n={len(chunks)} hit3={hit_at(ranks,golds):.3f} r10={recall(ranks,golds,10):.3f} r50={recall(ranks,golds,50):.3f} mrr={mrr(ranks,golds):.3f}", flush=True)
for s, st_ in [(0,0),(100,50),(200,100),(300,150),(500,250)]:
    run(s, st_)
