# Builds submission/index/ from the corpus. Build-time only, not part of the submission.
# usage: python build.py            (then: python tune.py, python local_check.py ../submission)
import glob, json, os, sys
from collections import Counter
import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(HERE, "..", "submission")
sys.path.insert(0, SUB)
from main import words, bm25_terms  # same text processing as at query time
from common import load_corpus, chunk_words

OUT = os.path.join(SUB, "index")
os.makedirs(OUT, exist_ok=True)
K1, B = 1.5, 0.9                 # doc-level BM25
CK1, CB = 1.2, 0.75              # chunk + title BM25
BM25_CHUNK, BM25_STRIDE = 300, 150
DENSE_CHUNK, DENSE_STRIDE = 300, 200
DENSE_DIMS = 256
MODEL = "minishlab/potion-retrieval-32M"

ids, texts, meta = load_corpus()
doc_words = [words(t) for t in texts]
doc_terms = [bm25_terms(w) for w in doc_words]
title_terms = [bm25_terms(words(m["title"] + " " + m.get("source_title", ""))) for m in meta]

chunk_terms, chunk_doc = [], []
for d, w in enumerate(doc_words):
    for c in chunk_words(" ".join(w), BM25_CHUNK, BM25_STRIDE):
        chunk_terms.append(bm25_terms(c.split())); chunk_doc.append(d)
chunk_doc = np.array(chunk_doc, np.int32)

vocab = sorted({t for ts in doc_terms + title_terms for t in ts})
tid = {t: i for i, t in enumerate(vocab)}


def bm25_csr(columns, k1, b):
    """columns: list of term lists. Returns term-major CSR of BM25 weights (idf * tf part), lucene idf."""
    n = len(columns)
    lens = np.array([len(c) for c in columns], np.float32); avg = lens.mean()
    rows, cols, tfs = [], [], []
    for j, c in enumerate(columns):
        for t, f in Counter(c).items():
            rows.append(tid[t]); cols.append(j); tfs.append(f)
    rows, cols, tfs = np.array(rows), np.array(cols), np.array(tfs, np.float32)
    df = np.bincount(rows, minlength=len(vocab))
    idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
    w = idf[rows] * tfs / (tfs + k1 * (1 - b + b * lens[cols] / avg))
    order = np.lexsort((cols, rows))
    indptr = np.zeros(len(vocab) + 1, np.int64); np.add.at(indptr, rows + 1, 1); indptr = np.cumsum(indptr)
    assert n < 65536
    return dict(indptr=indptr.astype(np.int32), cols=cols[order].astype(np.uint16), weights=w[order].astype(np.float16))


np.savez_compressed(os.path.join(OUT, "bm25_doc.npz"), **bm25_csr(doc_terms, K1, B))
np.savez_compressed(os.path.join(OUT, "bm25_title.npz"), **bm25_csr(title_terms, CK1, CB))
np.savez_compressed(os.path.join(OUT, "bm25_chunk.npz"), **bm25_csr(chunk_terms, CK1, CB))

# static embedding model: PCA to DENSE_DIMS, int8 per-row
snap = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{MODEL.replace('/', '--')}/snapshots/*"))[0]
E = load_file(os.path.join(snap, "model.safetensors"))["embeddings"].astype(np.float32)
mu = E.mean(0); U = np.linalg.svd(E - mu, full_matrices=False)[2]
T = (E - mu) @ U[:DENSE_DIMS].T
scale = (np.abs(T).max(1, keepdims=True) / 127).astype(np.float16)
table = np.round(T / scale.astype(np.float32)).astype(np.int8)
Tq = table.astype(np.float32) * scale.astype(np.float32)   # exactly what main.py rebuilds
tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
tok.save(os.path.join(OUT, "tokenizer.json"))

dchunks, dchunk_doc = [], []
for d, t in enumerate(texts):
    for c in chunk_words(t, DENSE_CHUNK, DENSE_STRIDE):
        dchunks.append(c); dchunk_doc.append(d)
dchunk_doc = np.array(dchunk_doc, np.int32)
V = np.zeros((len(dchunks), DENSE_DIMS), np.float32)
for i, e in enumerate(tok.encode_batch(dchunks, add_special_tokens=False)):
    if e.ids: V[i] = Tq[e.ids].mean(0)
V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
np.savez_compressed(os.path.join(OUT, "embeddings.npz"), table=table, scale=scale, chunk_vecs=V.astype(np.float16))

starts = lambda cd: np.searchsorted(cd, np.arange(len(ids))).astype(np.int32)  # chunks are in doc order
np.savez_compressed(os.path.join(OUT, "chunks.npz"), bm25_chunk_doc=chunk_doc.astype(np.uint16),
                    bm25_chunk_starts=starts(chunk_doc), dense_chunk_starts=starts(dchunk_doc))

counts = Counter(w for ws in doc_words for w in ws)
json.dump({w: n for w, n in counts.items() if n >= 2}, open(os.path.join(OUT, "word_counts.json"), "w"), separators=(",", ":"))
json.dump(vocab, open(os.path.join(OUT, "vocab.json"), "w"), separators=(",", ":"))
json.dump(ids, open(os.path.join(OUT, "doc_ids.json"), "w"))
cfg_path = os.path.join(OUT, "config.json")
if not os.path.exists(cfg_path):
    json.dump({"weights": {"doc": 1.0, "title": 0.3, "chunk": 0.7, "dense": 1.6}}, open(cfg_path, "w"), indent=1)
print("chunks: bm25", len(chunk_terms), "dense", len(dchunks), "vocab", len(vocab))
for f in sorted(os.listdir(OUT)):
    print(f"{os.path.getsize(os.path.join(OUT, f)) / 1e6:8.2f} MB  {f}")
