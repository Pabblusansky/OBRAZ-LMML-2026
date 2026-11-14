# Albania - evaluator local: ruleaza un detector pe datele de antrenament si
# calculeaza scorul oficial. Utilizare: python evaluate.py detector.py [-v]
import sys, importlib.util, time, math
import pandas as pd, numpy as np
# Calea detectorului (argumentul 1), implicit ../detector.py (versiunea finala)
path = sys.argv[1] if len(sys.argv) > 1 else '../detector.py'
# Importam fisierul ca modul, fara sa fie nevoie sa fie in sys.path
spec = importlib.util.spec_from_file_location('det', path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
# Datele de antrenament (relativ la work/)
base = '../files/'
x = pd.read_csv(base + 'train-input.csv'); y = pd.read_csv(base + 'train-output.csv')
# Trimitem citirile pe rand, ca in evaluarea reala, si verificam ca scorul e valid
d = m.Detector(); t = time.time(); s = []
for r in x.to_dict('records'):
    v = d.update(r); assert math.isfinite(v) and 0 <= v <= 1; s.append(v)
# Predictie = scor >= 0.5; comparam cu etichetele reale
p = np.array(s) >= 0.5; a = y.is_anomaly.values == 1
# Adevarat pozitive, fals pozitive, fals negative -> F1
tp = (p & a).sum(); fp = (p & ~a).sum(); fn = (~p & a).sum()
f1 = 2 * tp / (2 * tp + fp + fn)
# Recall separat pe fiecare tip de anomalie
rec = {k: p[y.anomaly_type.values == k].mean() for k in ['point', 'drift', 'collective']}
# Formula oficiala de scor
score = 100 * (0.8 * f1 + 0.05 * rec['point'] + 0.075 * rec['drift'] + 0.075 * rec['collective'])
print(f'time {time.time()-t:.2f}s tp {tp} fp {fp} fn {fn} F1 {f1:.3f} rec {rec} SCORE {score:.2f}')
# Cu -v afisam randurile clasificate gresit (pentru depanare)
if '-v' in sys.argv:
    y['p'] = p.astype(int); y['s'] = np.round(s, 2)
    print(y[(y.p == 1) != a].to_string())
