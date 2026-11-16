import numpy as np, time
from common import *; from onnx_enc import load
ids, texts, _ = load_corpus()
chunks = [c for t in texts for c in chunk_words(t, 200, 150)]
enc = load("int8", threads=6); t = time.time()
V = enc(chunks, bs=32); np.save("../runs/minilm_chunks_int8.npy", V); print("done", len(chunks), time.time() - t, flush=True)
