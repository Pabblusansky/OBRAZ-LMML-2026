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

BASE_N = 90      # cate citiri normale intra in mediana de referinta
LAG = 30          # intarzierea (pasi) dintre referinta si fereastra curenta
W_LONG = 6        # fereastra lunga: drift / anomalii colective
W_SHORT = 3       # fereastra scurta: salturi bruste
SPIKE_Z = 7.0     # prag z pentru varf izolat
T_LONG = 3.5      # prag z pentru fereastra lunga (drift)
T_SHORT = 5.0     # prag z pentru fereastra scurta (salt brusc)
T_COMB = 3.0      # prag pe canal cand >= 2 canale sunt de acord
WARMUP = 150      # primele 150 de citiri: doar invatam zgomotul, scor 0
MAX_RUN = 150     # o alarma mai lunga de atat = probabil un nou regim normal
VAR_ALPHA = 0.005  # viteza de actualizare a variantelor (medie exponentiala)
GAIN = 2.0        # panta sigmoidei: cat de repede trece scorul de la 0 la 1
END_RUN = 10      # dupa >= 10 pasi de alarma verificam daca anomalia s-a terminat
END_Z = 2.0       # fereastra scurta sub acest z = revenit la normal


def _to_float(value, default):
    """Converteste in float; valorile lipsa / ne-numerice / infinite -> valoarea implicita."""
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _sigmoid(logit):
    """Sigmoida 1/(1+e^-x), cu x limitat la [-50, 50] ca sa evitam overflow."""
    logit = max(-50.0, min(50.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def _median(values):
    """Mediana unei liste (robusta la valori extreme, spre deosebire de medie)."""
    sorted_vals = sorted(values)
    count = len(sorted_vals)
    if count == 0:
        return 0.0
    mid = count // 2
    return sorted_vals[mid] if count % 2 else 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


class Detector:
    def __init__(self):
        self.n_seen = 0                           # cate citiri am vazut
        self.pending = deque()               # citiri recente care inca nu pot intra in referinta
        self.baseline_buf = deque(maxlen=BASE_N)     # citiri normale pentru referinta
        self.recent = deque(maxlen=W_LONG)   # citiri recente curatate de varfuri
        self.raw_recent = deque(maxlen=5)           # ultimele 5 citiri brute (pentru mediana locala)
        self.last_clean = None                     # ultima citire curatata (pentru valori lipsa)
        self.var_long = None   # varianta abaterii ferestrei lungi
        self.var_short = None   # varianta abaterii ferestrei scurte
        self.var_point = None   # varianta abaterii citirii fata de mediana locala
        self.warmup_devs = []   # warmup deviations
        self.anom_run = 0                    # de cati pasi consecutivi dureaza alarma

    def update(self, reading: dict) -> float:
        """Primeste o citire si intoarce scorul de anomalie in [0, 1]."""
        # Valorile lipsa sunt inlocuite cu ultima valoare curata (sau 0 la inceput)
        prev_clean = self.last_clean if self.last_clean is not None else [0.0] * 6
        reading_vals = [_to_float(reading.get(col), prev_clean[ch]) for ch, col in enumerate(COLS)]
        self.n_seen += 1
        try:
            score = self._step(reading_vals)
        except Exception:
            # Orice eroare neasteptata nu trebuie sa opreasca evaluarea
            score = 0.0
        if not math.isfinite(score):
            score = 0.0
        # Scorul e mereu limitat la [0, 1]
        return float(min(1.0, max(0.0, score)))

    def _step(self, reading_vals):
        # --- varf izolat: citirea fata de mediana locala a ultimelor citiri
        # 1. DETECTORUL DE VARFURI: "ce ar trebui sa fie" = mediana ultimelor 5 citiri
        if len(self.raw_recent) >= 3:
            local_median = [_median([row[ch] for row in self.raw_recent]) for ch in range(6)]
        else:
            local_median = list(reading_vals)
        # Abaterea fata de mediana locala, pe fiecare canal
        dev_point = [reading_vals[ch] - local_median[ch] for ch in range(6)]
        # Impartita la abaterea standard a zgomotului -> scor z (dupa incalzire)
        if self.var_point is not None:
            z_point = [dev_point[ch] / math.sqrt(self.var_point[ch]) for ch in range(6)]
        else:
            z_point = [0.0] * 6
        z_point_max = max(abs(z) for z in z_point)
        spike = z_point_max > SPIKE_Z
        # Valoarea "curatata": canalele cu varf sunt inlocuite cu mediana locala,
        # ca varful sa nu strice ferestrele si referinta
        cleaned = [local_median[ch] if abs(z_point[ch]) > SPIKE_Z else reading_vals[ch] for ch in range(6)]
        self.recent.append(cleaned)
        self.raw_recent.append(reading_vals)
        self.last_clean = cleaned

        # --- abaterea ferestrelor recente fata de referinta intarziata
        # 2. REFERINTA: mediana citirilor normale intarziate
        if len(self.baseline_buf) >= 10:
            baseline = [_median([row[ch] for row in self.baseline_buf]) for ch in range(6)]
        else:
            # La inceput nu avem destule citiri in referinta: folosim si ce e in asteptare
            baseline_src = list(self.baseline_buf) + [row for row, _ in self.pending]
            baseline = [_median([row[ch] for row in baseline_src]) for ch in range(6)] if baseline_src else list(cleaned)
        recent_long = list(self.recent)
        recent_short = recent_long[-W_SHORT:]
        # dev_long = media ferestrei lungi - referinta; dev_short = media ferestrei scurte - referinta
        dev_long = [sum(row[ch] for row in recent_long) / len(recent_long) - baseline[ch] for ch in range(6)]
        dev_short = [sum(row[ch] for row in recent_short) / len(recent_short) - baseline[ch] for ch in range(6)]

        # INCALZIREA: colectam abaterile ca sa estimam zgomotul, scor 0
        if self.n_seen <= WARMUP:
            self.warmup_devs.append((dev_point, dev_long, dev_short))
            if self.n_seen == WARMUP:
                self._init_vars()
            self._push(cleaned, True)
            return 0.0

        # 3. SCORURI Z pentru ferestre
        z_long = [dev_long[ch] / math.sqrt(self.var_long[ch]) for ch in range(6)]
        z_short = [dev_short[ch] / math.sqrt(self.var_short[ch]) for ch in range(6)]
        # Sortate descrescator: abs_z_long[0] = cel mai abatut canal, abs_z_long[1] = al doilea
        abs_z_long = sorted((abs(z) for z in z_long), reverse=True)
        abs_z_short = sorted((abs(z) for z in z_short), reverse=True)
        # Fiecare detector: sigmoida(GAIN * (z - prag)) -> ~0 sub prag, ~1 peste
        s_point = _sigmoid(GAIN * (z_point_max - SPIKE_Z))          # varf izolat
        s_long = _sigmoid(GAIN * (abs_z_long[0] - T_LONG))            # drift pe un canal
        s_short = _sigmoid(GAIN * (abs_z_short[0] - T_SHORT))         # salt brusc pe un canal
        s_comb = _sigmoid(GAIN * (abs_z_long[1] - T_COMB))            # >= 2 canale moderat abatute (colectiv)
        s_comb2 = _sigmoid(GAIN * (abs_z_short[1] - T_COMB - 1.0))    # la fel, pe fereastra scurta
        # Scorul final = cel mai convins detector
        score = max(s_point, s_long, s_short, s_comb, s_comb2)
        # revenire brusca la normal dupa o alarma lunga: fereastra lunga inca "tine minte"
        # anomalia, dar ultimele citiri sunt din nou la nivelul referintei
        if self.anom_run >= END_RUN and abs_z_short[0] < END_Z and s_point < 0.5:
            score = min(score, 0.3)
        # Pentru depanare: ultimele scoruri partiale
        self.debug_scores = (s_point, s_long, s_short, s_comb, s_comb2)

        # is_normal = citirea e considerata normala (poate intra in referinta)
        is_normal = score < 0.5
        if is_normal:
            self.anom_run = 0
        else:
            self.anom_run += 1
            # Alarma foarte lunga -> acceptam noul nivel ca normal, sa nu ramanem blocati
            if self.anom_run > MAX_RUN:
                is_normal = True
        # actualizam variantele zgomotului (lent; atenuat cand citirea e semnalata)
        # 4. ACTUALIZAREA ZGOMOTULUI: medie exponentiala lenta a abaterilor la patrat,
        # de 20 de ori mai lenta in timpul unei alarme; contributia e plafonata la
        # 16 * varianta, ca un outlier sa nu umfle zgomotul
        alpha_eff = VAR_ALPHA * (1.0 if is_normal else 0.05)
        for ch in range(6):
            if not spike:
                self.var_point[ch] = (1 - alpha_eff) * self.var_point[ch] + alpha_eff * min(dev_point[ch] ** 2, 16 * self.var_point[ch])
            self.var_long[ch] = (1 - alpha_eff) * self.var_long[ch] + alpha_eff * min(dev_long[ch] ** 2, 16 * self.var_long[ch])
            self.var_short[ch] = (1 - alpha_eff) * self.var_short[ch] + alpha_eff * min(dev_short[ch] ** 2, 16 * self.var_short[ch])
        self._push(cleaned, is_normal)
        return score

    def _push(self, cleaned, is_normal):
        """Pune citirea in coada de asteptare; dupa LAG + W_LONG pasi intra in referinta
        (doar daca a fost normala). Astfel referinta e mereu "in urma" ferestrelor."""
        self.pending.append((cleaned, is_normal))
        while len(self.pending) > LAG + W_LONG:
            old_reading, was_normal = self.pending.popleft()
            if was_normal:
                self.baseline_buf.append(old_reading)

    def _init_vars(self):
        """La finalul incalzirii: estimam robust variantele zgomotului din abaterile colectate."""
        def robust_var(deviations):
            # MAD (mediana abaterilor absolute) * 1.4826 ~ abaterea standard pentru zgomot
            # normal, dar nu e afectata de cateva valori extreme
            abs_sorted = sorted(abs(dev) for dev in deviations)
            mad = abs_sorted[len(abs_sorted) // 2] * 1.4826
            return max(mad * mad, 1e-12)
        # Pentru ferestre sarim primii pasi, cand referinta nu era inca stabila
        warm_tail = self.warmup_devs[LAG + W_LONG:] or self.warmup_devs
        self.var_point = [robust_var([dev_tuple[0][ch] for dev_tuple in self.warmup_devs[5:]]) for ch in range(6)]
        self.var_long = [robust_var([dev_tuple[1][ch] for dev_tuple in warm_tail]) for ch in range(6)]
        self.var_short = [robust_var([dev_tuple[2][ch] for dev_tuple in warm_tail]) for ch in range(6)]
        self.warmup_devs = []
