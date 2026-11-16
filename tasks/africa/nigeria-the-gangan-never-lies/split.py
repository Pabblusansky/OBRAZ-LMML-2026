"""Split challenge.wav into 1 s chunks (wav + spectrogram each) and a contact sheet."""
import os, numpy as np, soundfile as sf
from scipy import signal
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x, sr = sf.read("data/challenge.wav", dtype="float32")
os.makedirs("out/chunks", exist_ok=True)
n = int(np.ceil(len(x) / sr))
fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 4), sharey=True)
for i in range(n):
    seg = x[i * sr:(i + 1) * sr]
    sf.write(f"out/chunks/chunk_{i:02d}.wav", seg, sr)
    f, t, S = signal.spectrogram(seg, sr, nperseg=512, noverlap=448)
    m = (f > 2500) & (f < 4300)
    axes[i].pcolormesh(t + i, f[m], 10 * np.log10(S[m] + 1e-12), shading="auto", cmap="gray")
    axes[i].set_title(f"{i}-{i+1}s")
plt.tight_layout(); plt.savefig("out/chunks_contact_sheet.png", dpi=70)
print(n, "chunks")
