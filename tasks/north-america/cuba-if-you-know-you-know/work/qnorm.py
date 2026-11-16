import re, unicodedata, collections
SP_STOP = set("la el los las de del donde parte capitulo que en y un una por para con como cuando cual sobre historia lo al se su sus es fue era muy mas entera toda todo cuenta dime".split())
SP_MAP = {"barcos":"ships","soldados":"soldiers","huelga":"strike","hijos":"sons","diosa":"goddess","iglesia":"church","cafe":"coffee","azucar":"sugar","tabaco":"tobacco","guerra":"war","isla":"island","ciudad":"city","rio":"river","esclavos":"slaves","ferrocarril":"railroad","campesinos":"peasants","lavanderas":"washerwomen","muerte":"death","rey":"king","puerto":"port","mujeres":"women","ninos":"children","dinero":"money","tierra":"land","agua":"water","fundadores":"founders","despensa":"pantry","despenza":"pantry"}
ABBR = {"abt":"about","w":"with","thru":"through","govt":"government","yr":"year","yrs":"years","tels":"tells","whch":"which","bc":"because"}
def norm(t):
    t = "".join(c for c in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", t)
def deletes(w):
    return {w[:i] + w[i+1:] for i in range(len(w))}
def lev(a, b):
    p = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        c = [i]
        for j, cb in enumerate(b, 1):
            c.append(min(p[j] + 1, c[j-1] + 1, p[j-1] + (ca != cb)))
        p = c
    return p[-1]
class Speller:
    def __init__(self, counts, minc=3):
        self.c = counts
        self.d = collections.defaultdict(list)
        for w, n in counts.items():
            if n >= minc and len(w) >= 3 and not w.isdigit():
                self.d[w].append(w)
                for x in deletes(w): self.d[x].append(w)
    def fix(self, w):
        if self.c.get(w, 0) >= 2 or len(w) < 4 or w.isdigit(): return w
        cands = set(self.d.get(w, []))
        for x in deletes(w): cands.update(self.d.get(x, []))
        best = None
        for v in cands:
            dist = lev(w, v)
            if dist <= 2:
                key = (dist, -self.c[v])
                if best is None or key < best[0]: best = (key, v)
        return best[1] if best else w
def clean(q, speller=None, spanish=True):
    out = []
    for w in norm(q):
        if spanish and w in SP_STOP: continue
        w = ABBR.get(w, w)
        if spanish: w = SP_MAP.get(w, w)
        if speller: w = speller.fix(w)
        out.append(w)
    return " ".join(out)
