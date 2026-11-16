import numpy as np, bm25s, Stemmer, collections, itertools
from common import *; from qnorm import *
ids, texts, meta = load_corpus(); Q = load_queries()
golds = [g for _, _, g in Q]
cnt = collections.Counter(); [cnt.update(norm(t)) for t in texts]
sp = Speller(cnt)
st = Stemmer.Stemmer("english")
variants = {"raw": [q for _, q, _ in Q], "sp_only": [clean(q, None) for _, q, _ in Q], "sp+spell": [clean(q, sp) for _, q, _ in Q]}
for (_, q, _), v in list(zip(Q, variants["sp+spell"]))[:0]: print(q, "->", v)
changed = [(q, v) for (_, q, _), v in zip(Q, variants["sp+spell"]) ]
corp = [" ".join(norm(t)) for t in texts]
ctok = bm25s.tokenize(corp, stopwords="en", stemmer=st, show_progress=False)
for k1, b in [(1.2,0.75),(1.5,0.75),(1.2,0.5),(0.9,0.4),(2.0,0.75),(1.5,0.9)]:
    r = bm25s.BM25(k1=k1, b=b); r.index(ctok, show_progress=False)
    for name, qs in variants.items():
        res, _ = r.retrieve(bm25s.tokenize(qs, stopwords="en", stemmer=st, show_progress=False), k=100, show_progress=False)
        ranks = [[ids[i] for i in row] for row in res]
        print(f"k1={k1} b={b} {name:9s} hit3={hit_at(ranks,golds):.3f} r10={recall(ranks,golds,10):.3f} r30={recall(ranks,golds,30):.3f} mrr={mrr(ranks,golds):.3f}")
