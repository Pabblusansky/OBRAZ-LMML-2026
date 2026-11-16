# Diagnostic: ce celule (cuvinte) din amprentele reale împing scorurile; vocabular invers bucket -> termeni din corpus
import csv, sys, numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, "files")
from featurize import tokens, bucket
from open_vault import CLASSES, load_sealed
rows = list(csv.DictReader(open("files/corpus.csv", encoding="utf-8")))
inv = defaultdict(Counter)
for r in rows:
    t = tokens(r["text"])
    for a in t: inv[bucket(a)][a] += 1
    for a, b2 in zip(t, t[1:]): inv[bucket(a+" "+b2)][a+" "+b2] += 1
def name(bk): return ",".join(w for w, _ in inv[bk].most_common(3)) or "?"
if __name__ == "__main__":
    Xs = load_sealed()
    m = np.load(sys.argv[1]); W, b = m["W"], m["b"]
    df = (Xs > 0).mean(0)
    print("most frequent sealed buckets:")
    for bk in np.argsort(-df)[:40]:
        print(f"  {df[bk]:.2f} {name(bk)[:40]:40s} W argmax={CLASSES[W[:,bk].argmax()]} {W[:,bk].max():.2f}")
    print("unseen-in-corpus share of sealed mass:", (Xs[:, [k for k in range(4096) if not inv[k]]]**2).sum(1).mean())
    S = Xs @ W.T + b; pred = S.argmax(1)
    for c in range(9):
        contrib = (Xs[pred == c] * W[c]).mean(0)
        top = np.argsort(-contrib)[:8]
        print(CLASSES[c], (pred==c).sum(), [name(k)[:20] for k in top])
