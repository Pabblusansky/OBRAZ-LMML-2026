import numpy as np, sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
d = np.load(sys.argv[1], allow_pickle=True)
E, T, y = d["image_embeddings"], d["text_embeddings"], d["labels"]
brands = ["nike","adidas","converse","puma","reebok","vans","asics"]
# zero-shot: which of 7 brand prompts is closest?
zs = (E @ T.T).argmax(1)
zs_names = np.array(brands)[zs]
print("zero-shot 7-brand: correct known brand", (zs_names == y).mean().round(3), "| wrongly sent to an unknown brand", np.isin(zs_names, brands[3:]).mean().round(3))
for b in brands: print("  ", b, (zs_names == b).sum())
for lab in ["nike","adidas","converse"]:
    m = y == lab; print("  true", lab, "->", dict(zip(*np.unique(zs_names[m], return_counts=True))))
# closed-set logistic regression, 5-fold
cv = StratifiedKFold(5, shuffle=True, random_state=0)
p = cross_val_predict(LogisticRegression(C=10, max_iter=3000), E, y, cv=cv, method="predict_proba")
cls = np.array(sorted(set(y)))
print("closed-set logistic CV accuracy", (cls[p.argmax(1)] == y).mean().round(3))
