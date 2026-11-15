# Rezolvare "Povești cu pene" (Moldova)
# 1) Fiecare citire e asociată păsării din registru de aceeași specie inelate în acel loc
#    (aproape mereu există exact un candidat) -> știm inscripția adevărată.
# 2) Din aceste perechi (adevărat, citit) estimăm, PENTRU FIECARE OBSERVATOR, o matrice de
#    confuzie P(citit | adevărat): fiecare observator are confuziile lui (S/Z, M/H, Q/D ...).
# 3) Citirile de barză albă fără niciun candidat în registru provin de la pasărea lipsă
#    (Prutul de Jos, Cahul). Pentru fiecare poziție alegem litera adevărată care maximizează
#    verosimilitatea tuturor citirilor, folosind matricea observatorului care a citit.
import sys, math, collections
import numpy as np, pandas as pd

import os
# fără argumente: folosim fișierele din files/ de lângă script
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
obs_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'observations.csv')
reg_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, 'ring_register.csv')
obs = pd.read_csv(obs_path); reg = pd.read_csv(reg_path)
r = obs[obs.ring_read.notna()].copy()
A = [chr(i) for i in range(65, 91)]

def cands(x):
    g = reg[reg.species == x.species]
    return list(g.ring[np.hypot(g.ringed_latitude - x.latitude, g.ringed_longitude - x.longitude) < 0.02])
r['cand'] = [cands(x) for _, x in r.iterrows()]

# matrici de confuzie: per observator (fiecare membru al listei) + globală
glob = collections.Counter(); per = collections.defaultdict(collections.Counter)
for _, x in r[r.cand.str.len() == 1].iterrows():
    for t, s in zip(x.cand[0], x.ring_read):
        if s == '?': continue
        glob[(t, s)] += 1
        for ob in x.observers.split('|'): per[ob][(t, s)] += 1

def P(t, s, obsrs, a=1.0, k=20.0):
    # amestec: contorul observatorilor, netezit spre matricea globală (care e netezită Laplace)
    pg = (glob[(t, s)] + a) / (sum(glob[(t, c)] for c in A) + 26 * a)
    c = collections.Counter()
    for ob in obsrs: c.update(per[ob])
    n = sum(c[(t, q)] for q in A)
    return (c[(t, s)] + k * pg) / (n + k)

miss = r[(r.cand.str.len() == 0) & (r.species == 'Ciconia ciconia')]
print(miss[miss.locality_id != 'L5819951'][['date','latitude','longitude','observers','ring_read']].to_string())
# păstrăm doar grupul repetat de la Prutul de Jos (L5819951); restul sunt citiri izolate ale altor berze rătăcite
miss = miss[miss.locality_id == 'L5819951']
print('citiri de barză albă fără inel în registru:', len(miss), miss.locality_id.unique())
word = ''
for p in range(6):
    sc = {t: sum(math.log(P(t, x.ring_read[p], x.observers.split('|')))
                 for _, x in miss.iterrows() if x.ring_read[p] != '?') for t in A}
    top = sorted(sc.items(), key=lambda v: -v[1])[:3]
    print('poziția', p + 1, [(t, round(v, 1)) for t, v in top]); word += top[0][0]
print('INEL:', word, '| în registru:', word in set(reg.ring))
print('FLAG: SIGMOID_' + word)
