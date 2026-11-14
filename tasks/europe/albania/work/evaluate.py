import sys, importlib.util, time, math
import pandas as pd, numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else 'detector.py'
spec = importlib.util.spec_from_file_location('det', path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
base = '/Users/dragoschisca/Desktop/LMML/Europa/albania/files/'
x = pd.read_csv(base + 'train-input.csv'); y = pd.read_csv(base + 'train-output.csv')
d = m.Detector(); t = time.time(); s = []
for r in x.to_dict('records'):
    v = d.update(r); assert math.isfinite(v) and 0 <= v <= 1; s.append(v)
p = np.array(s) >= 0.5; a = y.is_anomaly.values == 1
tp = (p & a).sum(); fp = (p & ~a).sum(); fn = (~p & a).sum()
f1 = 2 * tp / (2 * tp + fp + fn)
rec = {k: p[y.anomaly_type.values == k].mean() for k in ['point', 'drift', 'collective']}
score = 100 * (0.8 * f1 + 0.05 * rec['point'] + 0.075 * rec['drift'] + 0.075 * rec['collective'])
print(f'time {time.time()-t:.2f}s tp {tp} fp {fp} fn {fn} F1 {f1:.3f} rec {rec} SCORE {score:.2f}')
if '-v' in sys.argv:
    y['p'] = p.astype(int); y['s'] = np.round(s, 2)
    print(y[(y.p == 1) != a].to_string())
