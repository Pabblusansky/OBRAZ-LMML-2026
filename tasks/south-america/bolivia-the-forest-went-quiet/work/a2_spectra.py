import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.signal import welch, stft
X, rows = load()
fail = [i for i,r in enumerate(rows) if r["classification_status"]=="misclassified" and i!=29]
ref  = [i for i,r in enumerate(rows) if r["classification_status"]=="correct"]
f, S = welch(X, fs=16000, nperseg=4096)
L = 10*np.log10(S+1e-20)
# normalise each clip by its median level
Ln = L - np.median(L,axis=1,keepdims=True)
diff = Ln[fail].mean(0) - Ln[ref].mean(0)
top = np.argsort(diff)[::-1][:25]
print("freq bins where failures exceed refs (dB):")
for k in sorted(top): print(f"{f[k]:8.1f} Hz  +{diff[k]:.1f} dB   min-over-fails {(Ln[fail][:,k]-Ln[ref][:,k].mean()).min():.1f}")
# peakiness: per clip, prominent narrow peaks
from scipy.signal import find_peaks
print("\nnarrow peaks per failure clip (prominence>15dB):")
for i in fail+[29]+ref[:6]:
    pk,pr = find_peaks(L[i], prominence=15)
    print(i, rows[i]["classification_status"][:4], rows[i]["label"][:5], [round(f[p]) for p in pk[np.argsort(pr['prominences'])[::-1][:8]]])
