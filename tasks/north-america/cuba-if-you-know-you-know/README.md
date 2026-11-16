# If you know, you know! (Zemí)

| Field | Value |
|---|---|
| Country | Cuba |
| Difficulty | Hard |
| Category | retrieval / search under size + latency limits |
| Solved by | Pavel |
| Status | solved |
| Flag | `SIGMOID_...` (fill in after the platform accepts it) |

## 60-second version

For each question, return the 3 most likely of 1552 Cuba chapters/articles, under four
limits: Hit@3 ≥ 0.80, ≤ 53 MB, p95 ≤ 75 ms on one thread, load ≤ 10 s. The baseline
failed on size (80 MB) and accuracy (0.709). My retriever adds up four scores: BM25
over whole documents, BM25 over titles, BM25 over 300-word chunks, and a compressed
static embedding model (potion-retrieval-32M, PCA to 256 dims, int8). Queries are
cleaned first: Spanish filler removed, typos snapped to corpus words. On the 268
public questions: **Hit@3 0.873, 41.1 MB, 4.4 ms p95, 1.2 s load**. Cross-validated
estimate for unseen questions: **0.860** (worst split 0.821).

## Task

The questions are noisy on purpose: typos (`agaisnt`, `soviot`), Spanish mixed in
(`la parte donde`, `huelga`), abbreviations (`abt`, `w/`), and heavy paraphrase
("speaking-by-wire service" = telephone). The answer is often a detail deep inside a
long chapter. The corpus is absent at grading time, so everything ships in `index/`.
The runtime has numpy, scipy, bm25s, model2vec, tokenizers, PyStemmer and onnxruntime.
There is no torch and no network.

## Results (268 public questions, `local_check.py`)

| System | Hit@3 | Size | p95 |
|---|---|---|---|
| Baseline (bm25s + model2vec, RRF) | 0.709 | 80.3 MB ❌ | 3.2 ms |
| BM25, whole documents only | 0.776 | – | – |
| + query cleaning, k1=1.5, b=0.9 | 0.791 | – | – |
| Static embeddings alone | 0.668 | – | – |
| **Submitted: 4-signal fusion** | **0.873** | **41.1 MB** | **4.4 ms** |

The weights were chosen on the same 268 questions, so the honest number is
cross-validation: choose weights on half, test on the other half, 10 random splits →
mean **0.860**, worst **0.821**, best 0.896.

## How it works (`submission/main.py`)

1. **Clean the query.** Lowercase and strip accents. Drop Spanish filler (`la`, `de`,
   `donde`, `capitulo`, …). Map common Spanish words and short forms to English
   (`huelga` → strike, `abt` → about). Fix typos: a word the corpus has seen fewer than
   2 times becomes the most common corpus word within 2 edits (`agaisnt` → against).
2. **Score every document four ways.**
   - `doc`: BM25 over the whole text (k1=1.5, b=0.9).
   - `title`: BM25 over chapter title + book title. This helps "the whole story of X".
   - `chunk`: BM25 over 300-word chunks; a document scores as its best chunk. This
     finds details buried deep inside long chapters.
   - `dense`: cosine between the query and 300-word chunks with the static embedding
     model; best chunk again. This catches paraphrase.
3. **Fuse.** Z-score each signal over all 1552 documents, then take a weighted sum:
   doc 1.0, title 0.2, chunk 0.5, dense 1.6. Return the top k.

| index file | MB | content |
|---|---|---|
| `embeddings.npz` | 24.2 | token table (int8 + per-row scale), 19,596 chunk vectors (float16) |
| `bm25_chunk.npz` | 11.0 | BM25 weights for 25,685 chunks, per term (CSR) |
| `bm25_doc.npz` | 3.2 | BM25 weights for 1552 documents |
| `tokenizer.json`, `vocab.json`, `word_counts.json`, rest | 2.7 | tokenizer, term list, speller dictionary, ids, weights |

Edge cases checked: empty query, only stopwords, only punctuation, only Spanish
filler, a 3000-character query, k=1 and k=10. None crash; all return k ids in 1–6 ms.

## What failed / what I learned

| Tried | Result |
|---|---|
| Chunk-only BM25 (100-word chunks) | 0.713: chunks too small, worse than whole documents |
| Repeating the title inside the text | no change; a separate title field works (+4 pts) |
| Static model at 128 dims | dense part 0.69 → 0.59; 256 dims costs about 1 pt |
| int8 embedding table | no measurable loss |
| BM25 k1=0.9, b=0.4 | 0.716: long chapters need strong length normalisation |
| Typo-fixed query into the dense model | slightly worse than raw, so dense gets the raw text |
| MiniLM-L6 int8 ONNX instead of static model | CV 0.879 vs 0.860: better, but 2 pts is inside the noise of 268 questions; not shipped |
| snowflake-arctic-embed-xs | 0.791 alone, best single model; not finished, parked |

## Files

| Path | Purpose |
|---|---|
| `submission/` | `main.py` + `index/` (submission folder) |
| `dist/zemi_v1_static.zip` | the file to upload; passes `local_check.py` as a zip |
| `work/build.py` | builds `submission/index/` from the corpus (build-time only) |
| `work/tune.py` | picks the fusion weights, prints the cross-validated Hit@3 |
| `work/exp_*.py` | experiments behind the tables above |
| `task_files/` | task download (gitignored) |

Rebuild: `.venv-build/bin/python work/build.py && .venv/bin/python work/tune.py`, then
`python task_files/task_cuba/local_check.py submission`.
