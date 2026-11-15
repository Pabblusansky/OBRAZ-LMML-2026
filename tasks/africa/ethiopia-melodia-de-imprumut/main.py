"""Detector de îmbinare (splice) într-o înregistrare de azmari.

Ideea: bucata lipită nu are clicuri sau salturi de volum, dar provine din alt
context -> are alt „amprentă” acustică: alt registru/tonalitate (f0), alt timbru
(profilul armonicelor, ex. armonici impare dominante), alt zgomot de fond.
Calculăm pe cadre de 50 ms un vector de trăsături, le standardizăm și căutăm
intervalul [a, b] a cărui medie diferă cel mai mult de restul fișierului
(un model cu două puncte de schimbare: afară / înăuntru / afară).
"""
import sys
import csv
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from scipy.ndimage import median_filter


def load(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:                       # stereo -> mono
        x = x.mean(axis=1)
    m = np.max(np.abs(x))
    if m > 0:
        x /= m
    return sr, x


def features(sr, x):
    hop = int(round(0.05 * sr))          # pas de 50 ms
    nfft = 1 << int(np.ceil(np.log2(0.37 * sr)))   # fereastră lungă -> rezoluție bună în frecvență
    f, t, Z = stft(x, sr, nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
    P = np.abs(Z) + 1e-12
    df = f[1] - f[0]
    t = t  # timpul de centru al cadrelor

    # candidați f0: 80..1200 Hz pe scară logaritmică
    cands = 80.0 * 2 ** (np.arange(0, np.log2(1200 / 80), 1 / 96))
    H = 6
    logP = np.log(P)
    # maxim local (±2 bin) pentru a tolera mici abateri de frecvență
    from scipy.ndimage import maximum_filter1d
    logPm = maximum_filter1d(logP, size=5, axis=0)

    def bins(fr):
        return np.clip(np.round(fr / df).astype(int), 0, len(f) - 1)

    # scor de sumă armonică pentru fiecare candidat f0 (vectorizat pe cadre)
    score = np.zeros((len(cands), P.shape[1]))
    for h in range(1, H + 1):
        score += logPm[bins(cands * h), :]
    # penalizăm sub-armonicele: candidatul f0/2 ar „prinde” și armonicele lui f0
    half = logPm[bins(cands * 0.5), :]
    score -= 0.5 * np.maximum(half - logPm[bins(cands), :], 0)
    f0 = cands[np.argmax(score, axis=0)]

    feats = []
    nh = 8
    hidx = np.stack([bins(f0 * h) for h in range(1, nh + 1)])       # (nh, cadre)
    cols = np.arange(P.shape[1])
    hamp = np.stack([np.exp(logPm[hidx[k], cols]) for k in range(nh)])
    hdb = 20 * np.log10(hamp / hamp.max(axis=0, keepdims=True))
    hdb = np.maximum(hdb, -60)

    energy = 10 * np.log10(np.sum(P ** 2, axis=0))
    total = np.sum(P ** 2, axis=0)
    harm = np.sum(hamp ** 2, axis=0)
    resid = 10 * np.log10(np.maximum(total - harm, 1e-20) / total)   # zgomot / energie neharmonică

    odd = np.sum(hamp[0::2] ** 2, axis=0)   # armonicele 1,3,5,7
    even = np.sum(hamp[1::2] ** 2, axis=0)  # armonicele 2,4,6,8
    oddeven = 10 * np.log10(odd / (even + 1e-20))

    centroid = np.sum(f[:, None] * P ** 2, axis=0) / total
    hf = 10 * np.log10(np.sum(P[f > min(6000, sr / 2 * 0.6)] ** 2, axis=0) / total + 1e-20)
    lf = 10 * np.log10(np.sum(P[f < 100] ** 2, axis=0) / total + 1e-20)

    feats = [np.log2(f0) * 12,              # înălțimea în semitonuri
             oddeven,
             np.log2(centroid / f0) * 12,   # strălucire relativă la f0
             resid, hf, lf]
    feats += list(hdb[1:])                  # profil armonic h2..h8
    F = np.stack(feats, axis=1)

    # păstrăm doar cadrele cu sunet (nu liniștea)
    voiced = energy > np.percentile(energy, 90) - 35
    return t, F, voiced


def detect(path):
    sr, x = load(path)
    t, F, voiced = features(sr, x)
    dur = len(x) / sr

    # netezire cu median pe ~0.5 s ca să eliminăm erorile de cadru
    F = median_filter(F, size=(11, 1), mode="nearest")
    # standardizare robustă (mediană / MAD) pe cadrele cu sunet
    Fv = F[voiced]
    med = np.median(Fv, axis=0)
    mad = np.median(np.abs(Fv - med), axis=0) * 1.4826 + 1e-6
    Z = (F - med) / mad
    Z = np.clip(Z, -8, 8)
    Z[~voiced] = 0.0

    # sume cumulative -> media oricărui interval în O(1)
    w = voiced.astype(float)
    cz = np.vstack([np.zeros(Z.shape[1]), np.cumsum(Z * w[:, None], axis=0)])
    cw = np.concatenate([[0], np.cumsum(w)])
    tot_z, tot_w = cz[-1], cw[-1]

    step = 2                               # grilă de 0.1 s
    n = len(t)
    best = (-np.inf, 0, 0)
    min_len, max_len = int(1.5 / 0.05), int(min(25.0, 0.6 * dur) / 0.05)
    for a in range(0, n, step):
        for b in range(a + min_len, min(n, a + max_len) + 1, step):
            wi = cw[b] - cw[a]
            wo = tot_w - wi
            if wi < 10 or wo < 10:
                continue
            mi = (cz[b] - cz[a]) / wi
            mo = (tot_z - (cz[b] - cz[a])) / wo
            # statistică de tip t^2 însumată pe trăsături (ponderată cu mărimile grupurilor)
            s = np.sum((mi - mo) ** 2) * wi * wo / (wi + wo)
            if s > best[0]:
                best = (s, a, b)
    _, a, b = best
    hop_t = t[1] - t[0]
    start = max(0.0, t[a] - hop_t / 2)
    end = min(dur, t[b - 1] + hop_t / 2)
    return start, end


def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # rulare fără argument (ex. din IDE): folosim fișierul de antrenare din files/
        import os
        input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "files", "train-input.wav")
        print(f"Niciun argument dat, folosesc {input_file}")
    start, end = detect(input_file)
    with open("detection.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_s", "end_s"])
        w.writerow([f"{start:.3f}", f"{end:.3f}"])
    print(f"{start:.3f},{end:.3f}")


if __name__ == "__main__":
    main()
