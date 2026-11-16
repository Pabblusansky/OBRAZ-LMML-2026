# Învățare transductivă: probabilități din modele de bază -> propagare pe graful kNN al rapoartelor sigilate
# (rapoarte cu vocabular comun au de obicei același hazard) -> W liniar care reproduce etichetele.
import sys, os, numpy as np
sys.path.insert(0, "files")
from open_vault import try_vault
Xs = np.load("Xs.npy").astype(np.float64)
mask = np.flatnonzero((Xs > 0).mean(0) > 0.12)

def softmax(S, T):
    S = S / T; S -= S.max(1, keepdims=True); E = np.exp(S); return E / E.sum(1, keepdims=True)

def base_probs(names, T=None):
    P = 0
    for n in names:
        m = np.load(n); S = Xs @ m["W"].T + m["b"]
        S = (S - S.mean(1, keepdims=True)); S /= S.std() + 1e-9      # aceeași scară pentru fiecare model
        P = P + softmax(S, 0.5)
    return P / len(names)

def knn_graph(k=10):
    Z = Xs.copy(); Z[:, mask] = 0
    df = (Xs > 0).mean(0)
    Z = Z * np.log(1.0 / (df + 1.0 / len(Xs)))          # ponderare idf: cuvintele rare (tematice) contează mai mult
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    G = Z @ Z.T; np.fill_diagonal(G, 0)
    A = np.zeros_like(G)
    idx = np.argsort(-G, 1)[:, :k]
    for i in range(len(G)): A[i, idx[i]] = G[i, idx[i]]
    A = np.maximum(A, A.T)
    d = A.sum(1); d[d == 0] = 1
    return A / np.sqrt(d[:, None] * d[None, :])

CMN = True   # normalizarea masei claselor: propagarea nu are voie să umfle o clasă (ex. flood)

def propagate(P0, S, alpha=0.8, it=50):
    F = P0.copy(); target = P0.sum(0)
    for _ in range(it):
        F = alpha * S @ F + (1 - alpha) * P0
        if CMN: F = F * (target / F.sum(0))
    F = F / F.sum(1, keepdims=True)
    return F

def fit_linear(F):
    """W care dă pe rapoartele sigilate scorurile log F (regresie ridge cu dim 4096 > n)."""
    T = np.log(F + 1e-4); T -= T.mean(1, keepdims=True)
    lam = 1e-3
    K = Xs @ Xs.T
    A = np.linalg.solve(K + lam * np.eye(len(K)), T)       # W = Xs^T A
    W = (Xs.T @ A).T
    return W, np.zeros(9)

def em_priors(P, it=100):
    """Adaptare la schimbarea priorilor (Saerens et al. 2002): estimăm frecvența claselor în rapoartele reale."""
    pi = np.full(P.shape[1], 1.0 / P.shape[1])
    for _ in range(it):
        Q = P * (pi * P.shape[1]); Q /= Q.sum(1, keepdims=True)
        pi = Q.mean(0)
    return pi, Q

if __name__ == "__main__":
    names = sys.argv[1].split(",")
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    import os
    CMN = os.environ.get("CMN", "1") == "1"
    P0 = base_probs(names)
    if os.environ.get("EM", "0") == "1":
        pi, P0 = em_priors(P0); print("EM priors", np.round(pi, 3))
    S = knn_graph()
    F = propagate(P0, S, alpha) if alpha > 0 else P0
    print("base preds", np.bincount(P0.argmax(1), minlength=9), "after", np.bincount(F.argmax(1), minlength=9),
          "changed", int((P0.argmax(1) != F.argmax(1)).sum()))
    W, b = fit_linear(F)
    Sc = Xs @ W.T
    print("fit reproduces", (Sc.argmax(1) == F.argmax(1)).mean())
    out = sys.argv[3] if len(sys.argv) > 3 else "mt.npz"
    np.savez(out, W=W, b=b)
    print("vault:", try_vault(Sc, verbose=False))
