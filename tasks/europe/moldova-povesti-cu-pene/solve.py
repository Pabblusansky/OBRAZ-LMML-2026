# Rezolvare "Povești cu pene" (Moldova) -> SIGMOID_ANSEMO
#
# Ideea:
# 1) Fiecare citire de inel se asociază cu pasărea din registru de aceeași specie inelată în locul
#    observației (aproape mereu exact un candidat) -> știm inscripția adevărată a citirii.
# 2) Literele se confundă doar în 6 grupuri vizuale de câte 4: AKRX HMNW BGSZ EFLP CDOQ IJTY.
#    În fiecare grup există exact 3 „schimburi” (perechi de litere) -> 4 hărți posibile:
#    identitate, σ1, σ2, σ3 (grupul lui Klein). Fiecare pasăre e citită printr-o hartă FIXĂ
#    (comună păsărilor inelate în același loc): consensul observatorilor = σ(inel adevărat),
#    nu inelul adevărat. Votul majoritar e deci o capcană.
# 3) Barza lipsă = singurul grup de citiri de Ciconia ciconia incompatibil cu orice barză din
#    registru (comparând „semnătura de grup”): Prutul de Jos, L5819951 (Cahul), 24 de citiri.
# 4) Consensul ei pe poziții este RHGLWQ. Inelul adevărat este σ(RHGLWQ) pentru harta σ a păsării.
#    Pentru a afla σ: literele minoritare (abaterile de la consens) conțin „scurgeri” ale
#    adevărului. Comparăm, pentru fiecare hartă, cât de des apare relația σ între consens și
#    abatere la barza noastră față de cât de des apare ca zgomot de fond la păsările citite corect.
#    Harta cea mai supra-reprezentată (îmbogățire maximă) dă adevărul.
import os, sys, collections
import numpy as np, pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
obs = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'observations.csv'))
reg = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else os.path.join(BASE, 'ring_register.csv'))

GROUPS = ['AKRX', 'HMNW', 'BGSZ', 'EFLP', 'CDOQ', 'IJTY']
GROUP_OF = {c: g for g in GROUPS for c in g}

def make_map(pairs):
    d = {c: c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    for p in pairs.split():
        d[p[0]], d[p[1]] = p[1], p[0]
    return d

# cele 4 hărți (descoperite din date: ex. Dubăsari A->R, M->W...; Ștefan Vodă A->K, G->B...)
MAPS = {'id': make_map(''),
        's1': make_map('AR KX BZ GS CD OQ EL FP HN MW IJ TY'),
        's2': make_map('AX KR BS GZ CQ DO EP FL HM NW IT JY'),
        's3': make_map('AK RX BG SZ CO DQ EF LP HW MN IY JT')}

def majority(strings):
    """consensul pe poziții (litera cea mai frecventă, ignorând '?')"""
    out = ''
    for col in zip(*strings):
        c = collections.Counter(ch for ch in col if ch != '?')
        out += c.most_common(1)[0][0] if c else '?'
    return out

def relation(a, b):
    """ce hartă duce litera a în litera b (None dacă sunt în grupuri diferite)"""
    return next((k for k, m in MAPS.items() if m[a] == b), None)

# --- 1) citiri -> pasărea din registru (aceeași specie, inelată la <= 0.02° de observație)
reads = obs[obs.ring_read.notna()].copy()
def candidates(x):
    g = reg[reg.species == x.species]
    return list(g.ring[np.hypot(g.ringed_latitude - x.latitude, g.ringed_longitude - x.longitude) < 0.02])
reads['cand'] = [candidates(x) for _, x in reads.iterrows()]
known = reads[reads.cand.str.len() == 1].copy()
known['true'] = known.cand.str[0]

# --- 2) harta fiecărei păsări din registru (categoria majoritară a literelor citite)
bird_map = {}
for t, g in known.groupby('true'):
    c = collections.Counter(relation(a, b) for rd in g.ring_read for a, b in zip(t, rd) if b != '?')
    c.pop(None, None)
    if sum(c.values()) >= 12:
        bird_map[t] = c.most_common(1)[0][0]
print('hărți pe păsări din registru:', collections.Counter(bird_map.values()))
maj_ok = sum(1 for t, g in known.groupby('true') if t in bird_map and majority(g.ring_read) == t)
print(f'votul majoritar reface inelul corect doar la {maj_ok}/{len(bird_map)} păsări -> capcană')

# --- 3) barza lipsă: citiri de barză albă incompatibile (pe grupuri) cu orice barză din registru
sig = lambda s: ''.join(GROUP_OF.get(c, '?') if c != '?' else '?' for c in s)
storks = reg[reg.species == 'Ciconia ciconia'].ring
def compatible(rd, ring):
    return all(a == '?' or GROUP_OF[a] == GROUP_OF[b] for a, b in zip(rd, ring))
cc = reads[reads.species == 'Ciconia ciconia']
orphan = cc[[not any(compatible(rd, s) for s in storks) for rd in cc.ring_read]]
print('citiri orfane:', len(orphan), 'locuri:', orphan.locality_id.unique().tolist())
lp = orphan

# --- 4a) consensul pe poziții
consensus = majority(lp.ring_read)
print('consens (= σ(inel adevărat)):', consensus)

# --- 4b) zgomotul de fond: la păsările citite corect (harta id), ce relație au abaterile?
bg = collections.Counter()
for t, g in known.groupby('true'):
    if bird_map.get(t) != 'id':
        continue
    for rd in g.ring_read:
        for a, b in zip(t, rd):
            if b != '?' and b != a:
                bg[relation(a, b)] += 1
bg.pop(None, None)
bg_tot = sum(bg.values())

# --- 4c) abaterile barzei noastre față de consens
dev = collections.Counter(relation(a, b) for rd in lp.ring_read for a, b in zip(consensus, rd)
                          if b not in ('?', a))
dev.pop(None, None)
dev_tot = sum(dev.values())

print('\nharta | abateri barză | zgomot fond | îmbogățire | inel candidat')
scores = {}
for k in ['s1', 's2', 's3']:
    obs_f, bg_f = dev[k] / dev_tot, bg[k] / bg_tot
    scores[k] = obs_f / bg_f
    cand = ''.join(MAPS[k][c] for c in consensus)
    print(f'{k:5} | {dev[k]:3} ({obs_f:.2f})   | {bg_f:.2f}        | x{scores[k]:.2f}     | {cand}')

best = max(scores, key=scores.get)
ring = ''.join(MAPS[best][c] for c in consensus)
print(f'\nharta barzei: {best}  ->  INEL: {ring}  | în registru: {ring in set(reg.ring)}')
print('FLAG: SIGMOID_' + ring)
