# Albania - "Arata a anomalie!": detector de anomalii in flux (streaming).
#
# Primim cate o citire pe minut (6 senzori) si intoarcem un scor in [0, 1];
# peste 0.50 citirea e anomalie. Nu vedem viitorul, deci totul e "online".
#
# IDEEA: comparam ce vedem ACUM cu un "normal" de referinta construit din trecut
# si masuram diferenta in unitati de zgomot (scor z). Trei familii de detectoare:
#   1. varf izolat (anomalie "point")   : citirea vs mediana ultimelor 5 citiri
#   2. deplasare lenta (anomalie "drift"): media ultimelor 6 citiri vs referinta
#   3. salt brusc / mai multi senzori (anomalie "collective"): media ultimelor 3
#      citiri vs referinta, si "al doilea cel mai mare canal" (>= 2 senzori de acord)
# Referinta = mediana a 90 de citiri normale, INTARZIATA cu 30+ pasi, ca un drift
# lent sa nu fie "inghitit" de referinta inainte sa fie detectat.
import math
from collections import deque

# Cele 6 coloane de senzori, in ordinea folosita peste tot (index 0..5)
COLS = ["temperature_c", "humidity_pct", "vibration_g", "acoustic_db",
        "air_pressure_kpa", "co2_ppm"]

BASE_N = 90      # readings used for the baseline median
LAG = 30          # gap between baseline window and current window
W_LONG = 6        # window for drifts / collective
W_SHORT = 3       # window for abrupt changes
SPIKE_Z = 7.0     # prag z pentru varf izolat
T_LONG = 3.5      # prag z pentru fereastra lunga (drift)
T_SHORT = 5.0     # prag z pentru fereastra scurta (salt brusc)
T_COMB = 3.0      # per-channel level when >=2 channels agree
WARMUP = 150      # primele 150 de citiri: doar invatam zgomotul, scor 0
MAX_RUN = 150     # o alarma mai lunga de atat = probabil un nou regim normal
VAR_ALPHA = 0.005  # viteza de actualizare a variantelor (medie exponentiala)
GAIN = 2.0        # panta sigmoidei: cat de repede trece scorul de la 0 la 1
END_RUN = 10      # dupa >= 10 pasi de alarma verificam daca anomalia s-a terminat
END_Z = 2.0       # fereastra scurta sub acest z = revenit la normal


def _f(v, default):
    """Converteste in float; valorile lipsa / ne-numerice / infinite -> valoarea implicita."""
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _sig(x):
    """Sigmoida 1/(1+e^-x), cu x limitat la [-50, 50] ca sa evitam overflow."""
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _median(a):
    """Mediana unei liste (robusta la valori extreme, spre deosebire de medie)."""
    s = sorted(a)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


class Detector:
    def __init__(self):
        self.n = 0                           # cate citiri am vazut
        self.pending = deque()               # recent readings not yet eligible for baseline
        self.base = deque(maxlen=BASE_N)     # normal readings for baseline
        self.recent = deque(maxlen=W_LONG)   # spike-cleaned recent readings
        self.raw = deque(maxlen=5)           # ultimele 5 citiri brute (pentru mediana locala)
        self.last = None                     # ultima citire curatata (pentru valori lipsa)
        self.vl = None   # variance of long-window deviation
        self.vs = None   # variance of short-window deviation
        self.vp = None   # variance of single reading vs local median
        self.warm = []   # warmup deviations
        self.anom_run = 0                    # de cati pasi consecutivi dureaza alarma

    def update(self, reading: dict) -> float:
        """Primeste o citire si intoarce scorul de anomalie in [0, 1]."""
        # Valorile lipsa sunt inlocuite cu ultima valoare curata (sau 0 la inceput)
        prev = self.last if self.last is not None else [0.0] * 6
        v = [_f(reading.get(c), prev[i]) for i, c in enumerate(COLS)]
        self.n += 1
        try:
            score = self._step(v)
        except Exception:
            # Orice eroare neasteptata nu trebuie sa opreasca evaluarea
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        # Scorul e mereu limitat la [0, 1]
        return float(min(1.0, max(0.0, score)))

    def _step(self, v):
        # --- point spike vs local median of recent cleaned values
        # 1. DETECTORUL DE VARFURI: "ce ar trebui sa fie" = mediana ultimelor 5 citiri
        if len(self.raw) >= 3:
            loc = [_median([r[i] for r in self.raw]) for i in range(6)]
        else:
            loc = list(v)
        # Abaterea fata de mediana locala, pe fiecare canal
        dp = [v[i] - loc[i] for i in range(6)]
        # Impartita la abaterea standard a zgomotului -> scor z (dupa incalzire)
        if self.vp is not None:
            zp = [dp[i] / math.sqrt(self.vp[i]) for i in range(6)]
        else:
            zp = [0.0] * 6
        zpmax = max(abs(a) for a in zp)
        spike = zpmax > SPIKE_Z
        # Valoarea "curatata": canalele cu varf sunt inlocuite cu mediana locala,
        # ca varful sa nu strice ferestrele si referinta
        clean = [loc[i] if abs(zp[i]) > SPIKE_Z else v[i] for i in range(6)]
        self.recent.append(clean)
        self.raw.append(v)
        self.last = clean

        # --- deviation of recent windows from lagged baseline
        # 2. REFERINTA: mediana citirilor normale intarziate
        if len(self.base) >= 10:
            b = [_median([r[i] for r in self.base]) for i in range(6)]
        else:
            # La inceput nu avem destule citiri in referinta: folosim si ce e in asteptare
            src = list(self.base) + [r for r, _ in self.pending]
            b = [_median([r[i] for r in src]) for i in range(6)] if src else list(clean)
        rl = list(self.recent)
        rs = rl[-W_SHORT:]
        # dl = media ferestrei lungi - referinta; ds = media ferestrei scurte - referinta
        dl = [sum(r[i] for r in rl) / len(rl) - b[i] for i in range(6)]
        ds = [sum(r[i] for r in rs) / len(rs) - b[i] for i in range(6)]

        # INCALZIREA: colectam abaterile ca sa estimam zgomotul, scor 0
        if self.n <= WARMUP:
            self.warm.append((dp, dl, ds))
            if self.n == WARMUP:
                self._init_vars()
            self._push(clean, True)
            return 0.0

        # 3. SCORURI Z pentru ferestre
        zl = [dl[i] / math.sqrt(self.vl[i]) for i in range(6)]
        zs = [ds[i] / math.sqrt(self.vs[i]) for i in range(6)]
        # Sortate descrescator: al[0] = cel mai abatut canal, al[1] = al doilea
        al = sorted((abs(a) for a in zl), reverse=True)
        as_ = sorted((abs(a) for a in zs), reverse=True)
        # Fiecare detector: sigmoida(GAIN * (z - prag)) -> ~0 sub prag, ~1 peste
        s_point = _sig(GAIN * (zpmax - SPIKE_Z))          # varf izolat
        s_long = _sig(GAIN * (al[0] - T_LONG))            # drift pe un canal
        s_short = _sig(GAIN * (as_[0] - T_SHORT))         # salt brusc pe un canal
        s_comb = _sig(GAIN * (al[1] - T_COMB))            # >= 2 canale moderat abatute (colectiv)
        s_comb2 = _sig(GAIN * (as_[1] - T_COMB - 1.0))    # la fel, pe fereastra scurta
        # Scorul final = cel mai convins detector
        score = max(s_point, s_long, s_short, s_comb, s_comb2)
        # abrupt return to normal after a long alarm: the long window still
        # remembers the anomaly, but the latest readings are back at baseline
        if self.anom_run >= END_RUN and as_[0] < END_Z and s_point < 0.5:
            score = min(score, 0.3)
        # Pentru depanare: ultimele scoruri partiale
        self.dbg = (s_point, s_long, s_short, s_comb, s_comb2)

        # ok = citirea e considerata normala (poate intra in referinta)
        ok = score < 0.5
        if ok:
            self.anom_run = 0
        else:
            self.anom_run += 1
            # Alarma foarte lunga -> acceptam noul nivel ca normal, sa nu ramanem blocati
            if self.anom_run > MAX_RUN:
                ok = True
        # update noise variances (slowly; damped when flagged)
        # 4. ACTUALIZAREA ZGOMOTULUI: medie exponentiala lenta a abaterilor la patrat,
        # de 20 de ori mai lenta in timpul unei alarme; contributia e plafonata la
        # 16 * varianta, ca un outlier sa nu umfle zgomotul
        a = VAR_ALPHA * (1.0 if ok else 0.05)
        for i in range(6):
            if not spike:
                self.vp[i] = (1 - a) * self.vp[i] + a * min(dp[i] ** 2, 16 * self.vp[i])
            self.vl[i] = (1 - a) * self.vl[i] + a * min(dl[i] ** 2, 16 * self.vl[i])
            self.vs[i] = (1 - a) * self.vs[i] + a * min(ds[i] ** 2, 16 * self.vs[i])
        self._push(clean, ok)
        return score

    def _push(self, clean, ok):
        """Pune citirea in coada de asteptare; dupa LAG + W_LONG pasi intra in referinta
        (doar daca a fost normala). Astfel referinta e mereu "in urma" ferestrelor."""
        self.pending.append((clean, ok))
        while len(self.pending) > LAG + W_LONG:
            r, good = self.pending.popleft()
            if good:
                self.base.append(r)

    def _init_vars(self):
        """La finalul incalzirii: estimam robust variantele zgomotului din abaterile colectate."""
        def robust_var(vals):
            # MAD (mediana abaterilor absolute) * 1.4826 ~ abaterea standard pentru zgomot
            # normal, dar nu e afectata de cateva valori extreme
            s = sorted(abs(x) for x in vals)
            mad = s[len(s) // 2] * 1.4826
            return max(mad * mad, 1e-12)
        # Pentru ferestre sarim primii pasi, cand referinta nu era inca stabila
        w = self.warm[LAG + W_LONG:] or self.warm
        self.vp = [robust_var([d[0][i] for d in self.warm[5:]]) for i in range(6)]
        self.vl = [robust_var([d[1][i] for d in w]) for i in range(6)]
        self.vs = [robust_var([d[2][i] for d in w]) for i in range(6)]
        self.warm = []
