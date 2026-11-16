# Ca try_vault, dar arată ce ușă (index) și ce fracție s-a deschis — doar pentru modelele noastre.
import sys, numpy as np
sys.path.insert(0, "files")
import open_vault as ov
Xs = np.load("Xs.npy")
def info(S):
    z = np.load("files/vault.npz"); p, n, k = int(z["p"]), int(z["n"]), int(z["k"])
    pred = S.argmax(1); part = np.partition(S, -2, axis=1)
    order = np.argsort(-(part[:, -1] - part[:, -2]), kind="stable")
    inv = np.zeros(n + 2, dtype=np.int64)
    for d in range(1, n + 2): inv[d] = pow(d, p - 2, p)
    xs = np.arange(1, n + 1, dtype=np.int64); out = []
    for d in range(len(z["tables"])):
        door = dict(tag=z["tags"][d], sealed_word=z["words"][d])
        ys = z["tables"][d][np.arange(n), pred].astype(np.int64)
        for share in ov.SHARES:
            m = int(n * share)
            if m < k + 2: break
            take = np.sort(order[:m]); f = ov._decode(xs[take], ys[take], k, p, inv)
            if f is None: continue
            agree = int((ov._evaluate(f, xs[take], p) == ys[take]).sum())
            w = ov._word(f, k, door) if 2 * agree >= m + k else None
            if w: out.append((d, share, agree, m, w)); break
    return out
if __name__ == "__main__":
    for n in sys.argv[1:]:
        m = np.load(n); print(n, info(Xs @ m["W"].T + m["b"]))
