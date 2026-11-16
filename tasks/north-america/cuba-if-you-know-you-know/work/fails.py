import numpy as np, bm25s, Stemmer, collections, json
from common import *; from qnorm import *
ids, texts, meta = load_corpus(); Q = load_queries(); M = {m["id"]: m for m in meta}
cnt = collections.Counter(); [cnt.update(norm(t)) for t in texts]; sp = Speller(cnt)
st = Stemmer.Stemmer("english")
r = bm25s.BM25(k1=1.5, b=0.9); r.index(bm25s.tokenize([" ".join(norm(t)) for t in texts], stopwords="en", stemmer=st, show_progress=False), show_progress=False)
qs = [clean(q, sp) for _, q, _ in Q]
res, _ = r.retrieve(bm25s.tokenize(qs, stopwords="en", stemmer=st, show_progress=False), k=100, show_progress=False)
for (qid, q, g), row in zip(Q, res):
    rk = [ids[i] for i in row]
    if g not in rk[:3]:
        pos = rk.index(g) + 1 if g in rk else -1
        print(f"[{pos}] {q}\n   gold={g} ({M[g]['title'][:50]}) top={[(x, M[x]['title'][:30]) for x in rk[:2]]}")
