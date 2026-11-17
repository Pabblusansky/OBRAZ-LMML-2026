import numpy as np, sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
d = np.load(sys.argv[1], allow_pickle=True)
E, F, T, y = d["image_embeddings"], d["flipped_embeddings"], d["text_embeddings"], d["labels"]
known = np.array(["nike","adidas","converse"])
yi = np.array([list(known).index(v) for v in y])
cv = StratifiedKFold(5, shuffle=True, random_state=0)
pred = np.zeros(len(y), int); prob = np.zeros(len(y)); knn = np.zeros(len(y)); zs_unk = np.zeros(len(y))
for tr, te in cv.split(E, yi):
    clf = LogisticRegression(C=10, max_iter=3000).fit(np.vstack([E[tr], F[tr]]), np.r_[yi[tr], yi[tr]])
    P = clf.predict_proba(E[te]); pred[te] = P.argmax(1); prob[te] = P.max(1)
    S = E[te] @ E[tr].T       # cosine similarity to every training image
    for k, i in enumerate(te):  # mean of top-5 similarities among training images of the PREDICTED class
        s = S[k][yi[tr] == pred[i]]; knn[i] = np.sort(s)[-5:].mean()
    Z = np.exp(100 * E[te] @ T.T); Z /= Z.sum(1, keepdims=True)
    zs_unk[te] = Z[:, 3:].sum(1) - Z[np.arange(len(te)), pred[te]]   # >0: an unknown brand beats the predicted brand
print("closed-set CV acc", (pred == yi).mean().round(3))
for name, score, hi_is_unknown in [("max prob", prob, False), ("kNN top5 sim", knn, False), ("zero-shot unk margin", zs_unk, True)]:
    q = np.quantile(score, [0.05, 0.08, 0.92, 0.95])
    print(f"{name}: min {score.min():.3f} 5%/8% {q[0]:.3f}/{q[1]:.3f} 92%/95% {q[2]:.3f}/{q[3]:.3f} max {score.max():.3f}")
# known accuracy if we reject the most 'strange' 7% by a combined rank
for frac in [0.05, 0.07, 0.10]:
    thr = np.quantile(knn, frac); keep = knn >= thr
    print(f"reject {frac:.0%} lowest kNN -> known acc {((pred == yi) & keep).mean():.3f}")
