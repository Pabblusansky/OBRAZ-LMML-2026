import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.signal import butter, sosfiltfilt
X, rows = load()
sos = butter(4, 100, 'low', fs=16000, output='sos')
for i,r in enumerate(rows):
    x=X[i].astype(np.float64); lo=sosfiltfilt(sos,x)
    E=(x**2).sum(); el=(lo**2).sum()
    # spectrum of low band
    F=np.abs(np.fft.rfft(lo)); fr=np.fft.rfftfreq(len(lo),1/16000)
    pk=fr[np.argsort(F)[::-1][:3]]
    # low band envelope over 8 segments
    seg=(lo.reshape(8,-1)**2).mean(1); seg=np.sqrt(seg)
    print(f"{i:2d} {r['classification_status'][:4]} {r['label'][:5]:5s} dc={x.mean():+.5f} low%={100*el/E:6.2f} lowrms={np.sqrt(el/len(x)):.4f} pk={pk.round(1)} env={np.round(seg/ (seg.max()+1e-12),2)}")
