from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict

import numpy as np
import Stemmer
from tokenizers import Tokenizer

# words BM25 ignores (same list as bm25s "en")
STOPWORDS = set("a an and are as at be but by for if in into is it no not of on or such that the their "
                "then there these they this to was will with".split())
# Spanish filler words the slips mix in ("la parte donde ...")
SPANISH_FILLER = set("la el los las de del donde parte capitulo que en y un una por para con como cuando cual "
                     "sobre historia lo al se su sus es fue era muy mas entera toda todo cuenta dime".split())
# common Spanish words -> English, and short forms -> full words
WORD_MAP = {"barcos": "ships", "soldados": "soldiers", "huelga": "strike", "hijos": "sons", "diosa": "goddess",
            "iglesia": "church", "cafe": "coffee", "azucar": "sugar", "tabaco": "tobacco", "guerra": "war",
            "isla": "island", "ciudad": "city", "rio": "river", "esclavos": "slaves", "ferrocarril": "railroad",
            "campesinos": "peasants", "lavanderas": "washerwomen", "muerte": "death", "rey": "king",
            "puerto": "port", "mujeres": "women", "ninos": "children", "dinero": "money", "tierra": "land",
            "agua": "water", "fundadores": "founders", "despensa": "pantry", "despenza": "pantry",
            "abt": "about", "w": "with", "thru": "through", "govt": "government", "yr": "year", "yrs": "years",
            "tels": "tells", "whch": "which", "bc": "because"}

stemmer = Stemmer.Stemmer("english")


def words(text: str) -> list[str]:
    # lower case, drop accents, keep letters and digits
    text = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", text)


def bm25_terms(word_list: list[str]) -> list[str]:
    # the terms BM25 works with: no stopwords, no 1-letter words, stemmed
    return stemmer.stemWords([w for w in word_list if len(w) > 1 and w not in STOPWORDS])


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Speller:
    """Fixes typos by snapping unknown words to the closest corpus word (up to 2 edits)."""

    def __init__(self, counts: dict[str, int]) -> None:
        self.counts = counts
        self.by_delete = defaultdict(list)  # word with one letter removed -> original words
        for w, n in counts.items():
            if n >= 3 and len(w) >= 3 and not w.isdigit():
                self.by_delete[w].append(w)
                for i in range(len(w)):
                    self.by_delete[w[:i] + w[i + 1:]].append(w)

    def fix(self, w: str) -> str:
        if self.counts.get(w, 0) >= 2 or len(w) < 4 or w.isdigit():
            return w
        cands = set(self.by_delete.get(w, []))
        for i in range(len(w)):
            cands.update(self.by_delete.get(w[:i] + w[i + 1:], []))
        best = None
        for c in cands:
            d = edit_distance(w, c)
            if d <= 2 and (best is None or (d, -self.counts[c]) < best[0]):
                best = ((d, -self.counts[c]), c)  # fewest edits, then most common word
        return best[1] if best else w


def clean_query(query: str, speller: Speller) -> list[str]:
    out = []
    for w in words(query):
        if w in SPANISH_FILLER:
            continue
        out.append(speller.fix(WORD_MAP.get(w, w)))
    return out


class Bm25Matrix:
    """Precomputed BM25 weights, stored per term (CSR): term -> (column ids, weights)."""

    def __init__(self, path: str, vocab: dict[str, int], n_cols: int) -> None:
        z = np.load(path)
        self.indptr, self.cols, self.weights = z["indptr"], z["cols"].astype(np.int64), z["weights"].astype(np.float32)
        self.vocab, self.n_cols = vocab, n_cols

    def scores(self, terms: list[str]) -> np.ndarray:
        rows = [self.vocab[t] for t in set(terms) if t in self.vocab]
        if not rows:
            return np.zeros(self.n_cols, np.float32)
        parts = [slice(self.indptr[r], self.indptr[r + 1]) for r in rows]
        cols = np.concatenate([self.cols[p] for p in parts])
        w = np.concatenate([self.weights[p] for p in parts])
        return np.bincount(cols, weights=w, minlength=self.n_cols).astype(np.float32)


def zscore(s: np.ndarray) -> np.ndarray:
    # put every signal on the same scale before adding them
    return (s - s.mean()) / (s.std() + 1e-9)


class Retriever:
    def __init__(self, index_dir: str) -> None:
        def path(name):
            return os.path.join(index_dir, name)

        with open(path("config.json"), encoding="utf-8") as f:
            self.cfg = json.load(f)
        with open(path("doc_ids.json"), encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        with open(path("vocab.json"), encoding="utf-8") as f:
            vocab = {t: i for i, t in enumerate(json.load(f))}
        with open(path("word_counts.json"), encoding="utf-8") as f:
            self.speller = Speller(json.load(f))
        n_docs = len(self.doc_ids)

        # keyword signals: whole document, document title, best 300-word chunk
        self.bm25_doc = Bm25Matrix(path("bm25_doc.npz"), vocab, n_docs)
        self.bm25_title = Bm25Matrix(path("bm25_title.npz"), vocab, n_docs)
        z = np.load(path("chunks.npz"))
        self.bm25_chunk = Bm25Matrix(path("bm25_chunk.npz"), vocab, len(z["bm25_chunk_doc"]))
        self.bm25_chunk_starts = z["bm25_chunk_starts"]  # first chunk of each document
        self.dense_starts = z["dense_chunk_starts"]

        # meaning signal: static embedding model (one vector per token, query = average)
        self.tokenizer = Tokenizer.from_file(path("tokenizer.json"))
        e = np.load(path("embeddings.npz"))
        self.token_vecs = e["table"].astype(np.float32) * e["scale"].astype(np.float32)  # int8 -> float
        self.chunk_vecs = e["chunk_vecs"].astype(np.float32)  # one normalised vector per chunk

    def embed(self, text: str) -> np.ndarray:
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if not ids:
            return np.zeros(self.token_vecs.shape[1], np.float32)
        v = self.token_vecs[ids].mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    def search(self, query: str, k: int = 3) -> list[str]:
        w = self.cfg["weights"]
        terms = bm25_terms(clean_query(query, self.speller))

        doc = self.bm25_doc.scores(terms)
        title = self.bm25_title.scores(terms)
        # a document scores as its best chunk
        chunk = np.maximum.reduceat(self.bm25_chunk.scores(terms), self.bm25_chunk_starts)
        dense = np.maximum.reduceat(self.chunk_vecs @ self.embed(query), self.dense_starts)

        total = (w["doc"] * zscore(doc) + w["title"] * zscore(title)
                 + w["chunk"] * zscore(chunk) + w["dense"] * zscore(dense))
        top = np.argpartition(-total, k)[:k]
        top = top[np.argsort(-total[top])]
        return [self.doc_ids[i] for i in top]
