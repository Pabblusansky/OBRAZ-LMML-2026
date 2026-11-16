# usage: exp_dense.py <model> [size stride]
import sys, os, time, numpy as np
from common import *
name = sys.argv[1]; size = int(sys.argv[2]) if len(sys.argv) > 2 else 200; stride = int(sys.argv[3]) if len(sys.argv) > 3 else 150
PFX = {"BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
       "Snowflake/snowflake-arctic-embed-xs": ("Represent this sentence for searching relevant passages: ", ""),
       "Snowflake/snowflake-arctic-embed-s": ("Represent this sentence for searching relevant passages: ", ""),
       "intfloat/e5-small-v2": ("query: ", "passage: "),
       "intfloat/e5-base-v2": ("query: ", "passage: ")}
qp, pp = PFX.get(name, ("", ""))
ids, texts, meta = load_corpus(); Q = load_queries()
golds = [g for _, _, g in Q]; qtexts = [q for _, q, _ in Q]
chunks, cdoc = [], []
for d, t in enumerate(texts):
    for c in chunk_words(t, size, stride): chunks.append(c); cdoc.append(d)
cdoc = np.array(cdoc)
tag = name.replace("/", "__") + f"_{size}_{stride}"
out = os.path.join(os.path.dirname(__file__), "..", "runs", "dense", tag); os.makedirs(out, exist_ok=True)
if name.startswith("minishlab/"):
    from model2vec import StaticModel
    m = StaticModel.from_pretrained(name)
    enc = lambda xs, **kw: m.encode(xs)
else:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device="cpu"); m.max_seq_length = 320
    enc = lambda xs, **kw: m.encode(xs, batch_size=32, show_progress_bar=kw.get("bar", False), convert_to_numpy=True)
fn = os.path.join(out, "chunks.npy")
t0 = time.time()
if os.path.exists(fn): V = np.load(fn)
else:
    V = enc([pp + c for c in chunks], bar=True).astype(np.float32); np.save(fn, V)
V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
np.save(os.path.join(out, "cdoc.npy"), cdoc)
t1 = time.time()
q = enc([qp + x for x in qtexts]).astype(np.float32); q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
np.save(os.path.join(out, "queries.npy"), q)
tq = (time.time() - t1) / len(qtexts) * 1000
S = q @ V.T
D = np.full((len(qtexts), len(ids)), -9.0, dtype=np.float32)
for j in range(len(ids)): pass
np.maximum.at(D.T, cdoc, S.T) if False else None
# doc score = max chunk score
order = np.argsort(cdoc, kind="stable"); starts = np.searchsorted(cdoc[order], np.arange(len(ids)))
D = np.maximum.reduceat(S[:, order], starts, axis=1)
np.save(os.path.join(out, "docscores.npy"), D)
ranks = [[ids[j] for j in np.argsort(-row)[:100]] for row in D]
print(f"{tag} chunks={len(chunks)} enc_s={t1-t0:.0f} q_ms={tq:.1f} hit3={hit_at(ranks,golds):.3f} r10={recall(ranks,golds,10):.3f} r50={recall(ranks,golds,50):.3f} mrr={mrr(ranks,golds):.3f}", flush=True)
