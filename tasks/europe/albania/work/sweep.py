# Albania - cautare pe grila a constantelor detectorului.
# Utilizare (din work/): python sweep.py "{'SPIKE_Z': [6, 7], 'T_LONG': [3, 3.5]}" [../detector.py]
# Pentru fiecare combinatie de valori, suprascrie constantele din modul, ruleaza
# detectorul pe train si afiseaza cele mai bune 10 combinatii dupa scorul oficial.
import importlib.util, sys, itertools
import pandas as pd, numpy as np
# Datele de antrenament si etichetele reale
x = pd.read_csv('../files/train-input.csv'); y = pd.read_csv('../files/train-output.csv')
recs = x.to_dict('records'); a = y.is_anomaly.values == 1; t = y.anomaly_type.values
def run(params):
    # Reincarcam modulul de fiecare data, ca setarile anterioare sa nu ramana
    spec = importlib.util.spec_from_file_location('det', sys.argv[2] if len(sys.argv)>2 else '../detector.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    # Suprascriem constantele globale (ex. m.SPIKE_Z = 6)
    for k, v in params.items(): setattr(m, k, v)
    # Rulam detectorul si calculam F1 + recall pe tipuri, ca in evaluate.py
    d = m.Detector(); p = np.array([d.update(r) >= 0.5 for r in recs])
    tp = (p & a).sum(); fp = (p & ~a).sum(); fn = (~p & a).sum(); f1 = 2*tp/(2*tp+fp+fn)
    r = {k: p[t == k].mean() for k in ['point','drift','collective']}
    return 100*(0.8*f1+0.05*r['point']+0.075*r['drift']+0.075*r['collective']), tp, fp, fn, r
# Grila vine ca text Python din linia de comanda (doar pentru uz local)
grid = eval(sys.argv[1])
keys = list(grid)
res = []
# Produsul cartezian = toate combinatiile de valori
for vals in itertools.product(*grid.values()):
    pr = dict(zip(keys, vals)); sc = run(pr); res.append((sc[0], pr, sc[1:]))
# Primele 10 dupa scor
for r in sorted(res, key=lambda r: -r[0])[:10]: print(round(r[0],2), r[1], r[2][:3], {k: round(v,2) for k,v in r[2][3].items()})
