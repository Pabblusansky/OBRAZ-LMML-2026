# Albania - evaluator local: ruleaza un detector pe datele de antrenament si
# calculeaza scorul oficial. Utilizare: python evaluate.py detector.py [-v]
import sys, importlib.util, time, math
import pandas as pd, numpy as np
# Calea detectorului (argumentul 1), implicit ../detector.py (versiunea finala)
path = sys.argv[1] if len(sys.argv) > 1 else '../detector.py'
# Importam fisierul ca modul, fara sa fie nevoie sa fie in sys.path
spec = importlib.util.spec_from_file_location('det', path); det_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(det_module)
# Datele de antrenament (relativ la work/)
files_dir = '../files/'
inputs = pd.read_csv(files_dir + 'train-input.csv'); labels = pd.read_csv(files_dir + 'train-output.csv')
# Trimitem citirile pe rand, ca in evaluarea reala, si verificam ca scorul e valid
detector = det_module.Detector(); t_start = time.time(); scores = []
for reading in inputs.to_dict('records'):
    score_val = detector.update(reading); assert math.isfinite(score_val) and 0 <= score_val <= 1; scores.append(score_val)
# Predictie = scor >= 0.5; comparam cu etichetele reale
pred_anom = np.array(scores) >= 0.5; true_anom = labels.is_anomaly.values == 1
# Adevarat pozitive, fals pozitive, fals negative -> F1
tp = (pred_anom & true_anom).sum(); fp = (pred_anom & ~true_anom).sum(); fn = (~pred_anom & true_anom).sum()
f1 = 2 * tp / (2 * tp + fp + fn)
# Recall separat pe fiecare tip de anomalie
recall_by_type = {anom_type: pred_anom[labels.anomaly_type.values == anom_type].mean() for anom_type in ['point', 'drift', 'collective']}
# Formula oficiala de scor
score = 100 * (0.8 * f1 + 0.05 * recall_by_type['point'] + 0.075 * recall_by_type['drift'] + 0.075 * recall_by_type['collective'])
print(f'time {time.time()-t_start:.2f}s tp {tp} fp {fp} fn {fn} F1 {f1:.3f} rec {recall_by_type} SCORE {score:.2f}')
# Cu -v afisam randurile clasificate gresit (pentru depanare)
if '-v' in sys.argv:
    labels['p'] = pred_anom.astype(int); labels['s'] = np.round(scores, 2)
    print(labels[(labels.p == 1) != true_anom].to_string())
