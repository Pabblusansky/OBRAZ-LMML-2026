"""Analyse challenge.wav: spectrogram + LSB extraction."""
import sys, numpy as np, soundfile as sf
from scipy import signal
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x, sr = sf.read("data/challenge.wav", dtype="int16")
print("sr", sr, "n", len(x), "dur", len(x)/sr)
f, t, S = signal.spectrogram(x.astype(float), sr, nperseg=1024, noverlap=896)
plt.figure(figsize=(20, 7)); plt.pcolormesh(t, f, 10*np.log10(S+1e-9), shading="auto", cmap="magma")
plt.tight_layout(); plt.savefig("out/spectrogram.png", dpi=80)
for nb in (1, 2):
    bits = (x.astype(np.int32) & ((1 << nb) - 1))
    b = np.unpackbits(bits.astype(np.uint8)[:, None], axis=1)[:, 8-nb:].reshape(-1)
    for order in ("big", "little"):
        by = np.packbits(b, bitorder=order)[:400].tobytes()
        print(nb, order, by[:120])

# zoom on the hidden text band
m = (f > 2500) & (f < 4300)
plt.figure(figsize=(20, 4)); plt.pcolormesh(t, f[m], 10*np.log10(S[m]+1e-9), shading="auto", cmap="gray")
plt.tight_layout(); plt.savefig("out/spectrogram_zoom.png", dpi=80)
