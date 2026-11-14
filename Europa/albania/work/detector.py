import math
from collections import deque

COLS = ["temperature_c", "humidity_pct", "vibration_g", "acoustic_db",
        "air_pressure_kpa", "co2_ppm"]
# rough per-sensor noise scale (used until enough data is seen)
PRIOR_STD = [0.37, 0.58, 0.010, 0.35, 0.10, 10.5]

HIST = 240        # history kept
LAG = 75          # most recent readings excluded from the baseline
W_LONG = 10       # window for drifts
W_SHORT = 3       # window for abrupt collective changes
SPIKE_Z = 6.0
T_LONG = 1.6      # threshold on mean deviation (in noise std units)
T_SHORT = 2.6
COMB_DIV = 2.2
WARMUP = 130
MAX_RUN = 120
VAR_ALPHA = 0.01


def _f(v, default):
    try:
        v = float(v)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _sig(x):
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _median(a):
    s = sorted(a)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


class Detector:
    def __init__(self):
        self.n = 0
        self.hist = deque(maxlen=HIST)   # (values, is_normal)
        self.recent = deque(maxlen=W_LONG)  # spike-cleaned values
        self.last = None
        # noise estimate from first differences (robust to slow trends)
        self.dvar = [2 * s * s for s in PRIOR_STD]
        self.anom_run = 0
        self.vl = [0.3 * s * s for s in PRIOR_STD]   # var of long-window deviation
        self.vs = [0.5 * s * s for s in PRIOR_STD]   # var of short-window deviation
        self.nv = 0

    def _baseline(self):
        h = list(self.hist)
        cand = h[:-LAG] if len(h) > LAG + 20 else h
        normal = [v for v, ok in cand if ok]
        if len(normal) < 20:
            normal = [v for v, _ in cand] if cand else []
        if not normal:
            return None
        return [_median([v[i] for v in normal]) for i in range(6)]

    def update(self, reading: dict) -> float:
        prev = self.last if self.last is not None else [0.0] * 6
        v = [_f(reading.get(c), prev[i]) for i, c in enumerate(COLS)]
        self.n += 1
        std = [math.sqrt(max(self.dvar[i] / 2.0, 1e-12)) for i in range(6)]

        if self.n < 15:
            self.hist.append((v, True))
            self.recent.append(v)
            if self.last is not None:
                for i in range(6):
                    d = v[i] - self.last[i]
                    self.dvar[i] = 0.9 * self.dvar[i] + 0.1 * d * d
            self.last = v
            return 0.0

        # point spike: compare against the median of the recent cleaned values
        loc = [_median([r[i] for r in self.recent]) for i in range(6)]
        zp = [(v[i] - loc[i]) / std[i] for i in range(6)]
        zpmax = max(abs(a) for a in zp)
        s_point = _sig(1.5 * (zpmax - SPIKE_Z))
        spike = zpmax > SPIKE_Z
        clean = [loc[i] if abs(zp[i]) > SPIKE_Z else v[i] for i in range(6)]
        self.recent.append(clean)

        base = self._baseline()
        rl = list(self.recent)
        rs = rl[-W_SHORT:]
        dl = [sum(r[i] for r in rl) / len(rl) - base[i] for i in range(6)]
        ds = [sum(r[i] for r in rs) / len(rs) - base[i] for i in range(6)]
        zl = [dl[i] / math.sqrt(self.vl[i]) for i in range(6)]
        zs = [ds[i] / math.sqrt(self.vs[i]) for i in range(6)]

        def comb(z):
            a = sorted((abs(x) for x in z), reverse=True)
            return max(a[0], math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2) / COMB_DIV)
        warm = self.n < WARMUP
        s_long = 0.0 if warm else _sig(3.0 * (comb(zl) - T_LONG))
        s_short = 0.0 if warm else _sig(3.0 * (comb(zs) - T_SHORT))
        score = max(s_point, s_long, s_short)

        ok = score < 0.5
        if ok:
            self.anom_run = 0
        else:
            self.anom_run += 1
            if self.anom_run > MAX_RUN:
                ok = True
        self.hist.append((clean, ok))
        if len(self.hist) > LAG + 20:
            self.nv += 1
            a = max(1.0 / self.nv, VAR_ALPHA)
            if not ok or s_long > 0.5 or s_short > 0.5:
                a *= 0.1
            for i in range(6):
                self.vl[i] = (1 - a) * self.vl[i] + a * min(dl[i] ** 2, 25 * self.vl[i])
                self.vs[i] = (1 - a) * self.vs[i] + a * min(ds[i] ** 2, 25 * self.vs[i])
        if not spike:
            for i in range(6):
                d = v[i] - self.last[i]
                d2 = min(d * d, 16 * self.dvar[i])
                self.dvar[i] = 0.995 * self.dvar[i] + 0.005 * d2
        self.last = clean
        if not math.isfinite(score):
            score = 0.0
        return float(min(1.0, max(0.0, score)))
