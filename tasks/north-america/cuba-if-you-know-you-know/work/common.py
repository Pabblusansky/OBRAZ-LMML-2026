import json, os, re, unicodedata
import numpy as np
TASK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "task_files", "task_cuba")
DATA = os.path.join(TASK, "data")

def load_corpus():
    meta = [json.loads(l) for l in open(os.path.join(DATA, "corpus_meta.jsonl"), encoding="utf-8")]
    ids = [m["id"] for m in meta]
    texts = [open(os.path.join(DATA, "corpus", i + ".txt"), encoding="utf-8").read() for i in ids]
    return ids, texts, meta

def load_queries():
    qs = [json.loads(l) for l in open(os.path.join(DATA, "sample-input.jsonl"), encoding="utf-8")]
    gold = {json.loads(l)["qid"]: json.loads(l)["doc_id"] for l in open(os.path.join(DATA, "sample-output.jsonl"), encoding="utf-8")}
    return [(q["qid"], q["question"], gold[q["qid"]]) for q in qs]

def hit_at(rankings, golds, k=3):
    return float(np.mean([g in r[:k] for r, g in zip(rankings, golds)]))

def mrr(rankings, golds, k=100):
    out = []
    for r, g in zip(rankings, golds):
        r = list(r[:k]); out.append(1 / (r.index(g) + 1) if g in r else 0)
    return float(np.mean(out))

def recall(rankings, golds, k):
    return float(np.mean([g in r[:k] for r, g in zip(rankings, golds)]))

def chunk_words(text, size, stride):
    w = text.split()
    if len(w) <= size: return [" ".join(w)]
    return [" ".join(w[i:i + size]) for i in range(0, max(1, len(w) - size + stride), stride)]
