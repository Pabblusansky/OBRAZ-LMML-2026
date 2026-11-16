# save dense doc scores for compressed potion variants: args dims size stride
import os, sys, numpy as np, glob
from safetensors.numpy import load_file
from tokenizers import Tokenizer
from common import *
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--minishlab--potion-retrieval-32M/snapshots/*"))[0]
E = load_file(os.path.join(snap, "model.safetensors"))["embeddings"].astype(np.float32)
tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
ids, texts, meta = load_corpus(); Q = load_queries()
dims, size, stride = map(int, sys.argv[1:4])
mu = E.mean(0); U = np.linalg.svd(E - mu, full_matrices=False)[2]
T = (E - mu) @ U[:dims].T if dims < E.shape[1] else E
s = np.abs(T).max(1, keepdims=True) / 127; T = np.round(T / s).astype(np.int8).astype(np.float32) * s
chunks, cdoc = [], []
for d, t in enumerate(texts):
    for c in chunk_words(t, size, stride): chunks.append(c); cdoc.append(d)
cdoc = np.array(cdoc); order = np.argsort(cdoc, kind="stable"); starts = np.searchsorted(cdoc[order], np.arange(len(ids)))
def enc(xs):
    out = np.zeros((len(xs), T.shape[1]), np.float32)
    for i, e in enumerate(tok.encode_batch(xs, add_special_tokens=False)):
        if e.ids: out[i] = T[e.ids].mean(0)
    return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
V = enc(chunks); q = enc([x for _, x, _ in Q])
D = np.maximum.reduceat((q @ V.T)[:, order], starts, axis=1)
np.save(f"../runs/potion_d{dims}_c{size}.npy", D); print(len(chunks), "chunks saved")
