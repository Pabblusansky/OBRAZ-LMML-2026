# Explorare: corpus, modelul inițial și seiful
import sys, csv, numpy as np
sys.path.insert(0, "files")
from featurize import featurize
from open_vault import CLASSES, load_sealed, try_vault
rows = list(csv.DictReader(open("files/corpus.csv", encoding="utf-8")))
from collections import Counter
print(len(rows), Counter(r["label"] for r in rows))
X = featurize([r["text"] for r in rows]); y = np.array([CLASSES.index(r["label"]) for r in rows])
np.save("X.npy", X); np.save("y.npy", y)
Xs = load_sealed(); np.save("Xs.npy", Xs)
m = np.load("files/first_model.npz"); W, b = m["W"], m["b"]
S = Xs @ W.T + b
print("first model preds on sealed:", Counter(S.argmax(1)))
print("train acc", ((X@W.T+b).argmax(1)==y).mean())
print("nnz sealed", (Xs>0).sum(1).mean(), "nnz corpus", (X>0).sum(1).mean())
print(try_vault(S))
