# compress the potion table (PCA dims, int8) and measure dense + fused hit@3
import os, sys, numpy as np, collections, glob
from safetensors.numpy import load_file
from tokenizers import Tokenizer
from common import *; from qnorm import *
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--minishlab--potion-retrieval-32M/snapshots/*"))[0]
E = load_file(os.path.join(snap, "model.safetensors"))["embeddings"].astype(np.float32)
tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
ids, texts, meta = load_corpus(); Q = load_queries()
golds = np.array([ids.index(g) for _, _, g in Q])
cnt = collections.Counter(); [cnt.update(norm(t)) for t in texts]; sp = Speller(cnt)
size, stride = int(sys.argv[1]) if len(sys.argv) > 1 else 200, int(sys.argv[2]) if len(sys.argv) > 2 else 150
chunks, cdoc = [], []
for d, t in enumerate(texts):
    for c in chunk_words(t, size, stride): chunks.append(c); cdoc.append(d)
cdoc = np.array(cdoc); order = np.argsort(cdoc, kind="stable"); starts = np.searchsorted(cdoc[order], np.arange(len(ids)))
def enc(T, xs):
    out = np.zeros((len(xs), T.shape[1]), np.float32)
    for i, e in enumerate(tok.encode_batch(xs, add_special_tokens=False)):
        if e.ids: out[i] = T[e.ids].mean(0)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
import bm25s, Stemmer
st = Stemmer.Stemmer("english")
qclean = [clean(q, sp) for _, q, _ in Q]; qraw = [q for _, q, _ in Q]
r = bm25s.BM25(k1=1.5, b=0.9); r.index(bm25s.tokenize([" ".join(norm(t)) for t in texts], stopwords="en", stemmer=st, show_progress=False), show_progress=False)
res, sc = r.retrieve(bm25s.tokenize(qclean, stopwords="en", stemmer=st, show_progress=False), k=len(ids), show_progress=False)
B = np.zeros((len(Q), len(ids)), np.float32)
for i in range(len(Q)): B[i, res[i]] = sc[i]
zs = lambda S: (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
def h3(S): top = np.argsort(-S, 1)[:, :3]; return float(np.mean([g in t for g, t in zip(golds, top)]))
Bz = zs(B)
mu = E.mean(0)
U = np.linalg.svd(E - mu, full_matrices=False)[2]
for dims in [512, 256, 128]:
    T = (E - mu) @ U[:dims].T if dims < 512 else E.copy()
    for q8 in [False, True]:
        Tq = T
        if q8:
            s = np.abs(T).max(1, keepdims=True) / 127; Tq = (np.round(T / s).astype(np.int8)).astype(np.float32) * s
        V = enc(Tq, chunks)
        for qname, qs in [("raw", qraw), ("clean", qclean)]:
            q = enc(Tq, qs); S = q @ V.T
            D = np.maximum.reduceat(S[:, order], starts, axis=1)
            print(f"dims={dims} int8={q8} q={qname} dense={h3(D):.3f} " + " ".join(f"w{w}={h3(Bz + w*zs(D)):.3f}" for w in [0.7,1.0,1.3,1.6]), flush=True)
