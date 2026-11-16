# Acuratețea modelelor pe rapoartele noastre inventate (validare în stil „uman")
import sys, numpy as np
sys.path.insert(0, "files"); sys.path.insert(0, ".")
from featurize import featurize
from open_vault import CLASSES
from my_reports import REPORTS
Xv = featurize([t for _, t in REPORTS]); yv = np.array([CLASSES.index(c) for c, _ in REPORTS])
for n in sys.argv[1:]:
    m = np.load(n); p = (Xv @ m["W"].T + m["b"]).argmax(1)
    wrong = [(CLASSES[yv[i]], CLASSES[p[i]]) for i in np.flatnonzero(p != yv)]
    print(f"{n:22s} acc {np.mean(p == yv):.3f}", wrong[:12])
