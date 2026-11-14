import math
from collections import deque

COLS = ["temperature_c", "humidity_pct", "vibration_g", "acoustic_db",
        "air_pressure_kpa", "co2_ppm"]

BASE_N = 90      # readings used for the baseline median
LAG = 30          # gap between baseline window and current window
W_LONG = 6        # window for drifts / collective
W_SHORT = 3       # window for abrupt changes
SPIKE_Z = 7.0
T_LONG = 3.5
T_SHORT = 5.0
T_COMB = 3.0      # per-channel level when >=2 channels agree
WARMUP = 150
MAX_RUN = 150
VAR_ALPHA = 0.005
GAIN = 2.0
END_RUN = 10
END_Z = 2.0


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
        self.pending = deque()               # recent readings not yet eligible for baseline
        self.base = deque(maxlen=BASE_N)     # normal readings for baseline
        self.recent = deque(maxlen=W_LONG)   # spike-cleaned recent readings
        self.raw = deque(maxlen=5)
        self.last = None
        self.vl = None   # variance of long-window deviation
        self.vs = None   # variance of short-window deviation
        self.vp = None   # variance of single reading vs local median
        self.warm = []   # warmup deviations
        self.anom_run = 0

    def update(self, reading: dict) -> float:
        prev = self.last if self.last is not None else [0.0] * 6
        v = [_f(reading.get(c), prev[i]) for i, c in enumerate(COLS)]
        self.n += 1
        try:
            score = self._step(v)
        except Exception:
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        return float(min(1.0, max(0.0, score)))

    def _step(self, v):
        # --- point spike vs local median of recent cleaned values
        if len(self.raw) >= 3:
            loc = [_median([r[i] for r in self.raw]) for i in range(6)]
        else:
            loc = list(v)
        dp = [v[i] - loc[i] for i in range(6)]
        if self.vp is not None:
            zp = [dp[i] / math.sqrt(self.vp[i]) for i in range(6)]
        else:
            zp = [0.0] * 6
        zpmax = max(abs(a) for a in zp)
        spike = zpmax > SPIKE_Z
        clean = [loc[i] if abs(zp[i]) > SPIKE_Z else v[i] for i in range(6)]
        self.recent.append(clean)
        self.raw.append(v)
        self.last = clean

        # --- deviation of recent windows from lagged baseline
        if len(self.base) >= 10:
            b = [_median([r[i] for r in self.base]) for i in range(6)]
        else:
            src = list(self.base) + [r for r, _ in self.pending]
            b = [_median([r[i] for r in src]) for i in range(6)] if src else list(clean)
        rl = list(self.recent)
        rs = rl[-W_SHORT:]
        dl = [sum(r[i] for r in rl) / len(rl) - b[i] for i in range(6)]
        ds = [sum(r[i] for r in rs) / len(rs) - b[i] for i in range(6)]

        if self.n <= WARMUP:
            self.warm.append((dp, dl, ds))
            if self.n == WARMUP:
                self._init_vars()
            self._push(clean, True)
            return 0.0

        zl = [dl[i] / math.sqrt(self.vl[i]) for i in range(6)]
        zs = [ds[i] / math.sqrt(self.vs[i]) for i in range(6)]
        al = sorted((abs(a) for a in zl), reverse=True)
        as_ = sorted((abs(a) for a in zs), reverse=True)
        s_point = _sig(GAIN * (zpmax - SPIKE_Z))
        s_long = _sig(GAIN * (al[0] - T_LONG))
        s_short = _sig(GAIN * (as_[0] - T_SHORT))
        s_comb = _sig(GAIN * (al[1] - T_COMB))
        s_comb2 = _sig(GAIN * (as_[1] - T_COMB - 1.0))
        score = max(s_point, s_long, s_short, s_comb, s_comb2)
        # abrupt return to normal after a long alarm: the long window still
        # remembers the anomaly, but the latest readings are back at baseline
        if self.anom_run >= END_RUN and as_[0] < END_Z and s_point < 0.5:
            score = min(score, 0.3)
        self.dbg = (s_point, s_long, s_short, s_comb, s_comb2)

        ok = score < 0.5
        if ok:
            self.anom_run = 0
        else:
            self.anom_run += 1
            if self.anom_run > MAX_RUN:
                ok = True
        # update noise variances (slowly; damped when flagged)
        a = VAR_ALPHA * (1.0 if ok else 0.05)
        for i in range(6):
            if not spike:
                self.vp[i] = (1 - a) * self.vp[i] + a * min(dp[i] ** 2, 16 * self.vp[i])
            self.vl[i] = (1 - a) * self.vl[i] + a * min(dl[i] ** 2, 16 * self.vl[i])
            self.vs[i] = (1 - a) * self.vs[i] + a * min(ds[i] ** 2, 16 * self.vs[i])
        self._push(clean, ok)
        return score

    def _push(self, clean, ok):
        self.pending.append((clean, ok))
        while len(self.pending) > LAG + W_LONG:
            r, good = self.pending.popleft()
            if good:
                self.base.append(r)

    def _init_vars(self):
        def robust_var(vals):
            s = sorted(abs(x) for x in vals)
            mad = s[len(s) // 2] * 1.4826
            return max(mad * mad, 1e-12)
        w = self.warm[LAG + W_LONG:] or self.warm
        self.vp = [robust_var([d[0][i] for d in self.warm[5:]]) for i in range(6)]
        self.vl = [robust_var([d[1][i] for d in w]) for i in range(6)]
        self.vs = [robust_var([d[2][i] for d in w]) for i in range(6)]
        self.warm = []
