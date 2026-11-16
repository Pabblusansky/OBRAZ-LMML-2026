# Albania - cautare pe grila a constantelor detectorului.
# Utilizare (din work/): python sweep.py "{'SPIKE_Z': [6, 7], 'T_LONG': [3, 3.5]}" [../detector.py]
# Pentru fiecare combinatie de valori, suprascrie constantele din modul, ruleaza
# detectorul pe train si afiseaza cele mai bune 10 combinatii dupa scorul oficial.
import importlib.util, sys, itertools
import pandas as pd, numpy as np
# Datele de antrenament si etichetele reale
inputs = pd.read_csv('../files/train-input.csv'); labels = pd.read_csv('../files/train-output.csv')
records = inputs.to_dict('records'); true_anom = labels.is_anomaly.values == 1; anom_types = labels.anomaly_type.values
def run(params):
    # Reincarcam modulul de fiecare data, ca setarile anterioare sa nu ramana
    spec = importlib.util.spec_from_file_location('det', sys.argv[2] if len(sys.argv)>2 else '../detector.py'); det_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(det_module)
    # Suprascriem constantele globale (ex. det_module.SPIKE_Z = 6)
    for param_name, param_val in params.items(): setattr(det_module, param_name, param_val)
    # Rulam detectorul si calculam F1 + recall pe tipuri, ca in evaluate.py
    detector = det_module.Detector(); pred_anom = np.array([detector.update(recall_by_type) >= 0.5 for recall_by_type in records])
    tp = (pred_anom & true_anom).sum(); fp = (pred_anom & ~true_anom).sum(); fn = (~pred_anom & true_anom).sum(); f1 = 2*tp/(2*tp+fp+fn)
    recall_by_type = {anom_type: pred_anom[anom_types == anom_type].mean() for anom_type in ['point','drift','collective']}
    return 100*(0.8*f1+0.05*recall_by_type['point']+0.075*recall_by_type['drift']+0.075*recall_by_type['collective']), tp, fp, fn, recall_by_type
# Grila vine ca text Python din linia de comanda (doar pentru uz local)
grid = eval(sys.argv[1])
keys = list(grid)
results = []
# Produsul cartezian = toate combinatiile de valori
for combo in itertools.product(*grid.values()):
    params_combo = dict(zip(keys, combo)); run_result = run(params_combo); results.append((run_result[0], params_combo, run_result[1:]))
# Primele 10 dupa scor
for entry in sorted(results, key=lambda entry: -entry[0])[:10]: print(round(entry[0],2), entry[1], entry[2][:3], {type_name: round(recall_val,2) for type_name,recall_val in entry[2][3].items()})
