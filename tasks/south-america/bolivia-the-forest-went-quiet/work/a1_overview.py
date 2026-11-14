import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import hashlib
X, rows = load()
P = predict_probabilities(model, X)
for i,(r,p) in enumerate(zip(rows,P)):
    h = hashlib.sha256(open(T/r["path"],"rb").read()).hexdigest()==r["sha256"]
    print(f"{i:2d} {r['clip_id']} {r['classification_status']:13s} {r['label']:8s} pred={CLASS_NAMES[p.argmax()]:8s} "+" ".join(f"{v:.2f}" for v in p), "rms=%.4f peak=%.3f"%(np.sqrt((X[i]**2).mean()),abs(X[i]).max()), "hashok" if h else "HASHBAD")
d=np.load(T/"train-input.npz"); print({k:(d[k].shape,d[k].dtype) for k in d}); print(d["clip_ids"][:5])
Pt=predict_probabilities(model,d["audio"]); print("train preds", np.bincount(Pt.argmax(1),minlength=4))
