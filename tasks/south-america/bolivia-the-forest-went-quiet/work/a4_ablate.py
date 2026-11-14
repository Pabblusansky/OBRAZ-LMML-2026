import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.signal import butter, sosfiltfilt
X, rows = load()
fail = [i for i,r in enumerate(rows) if r["classification_status"]=="misclassified" and i!=29]
edges=[0,50,100,200,400,700,1000,1500,2000,3000,4000,5000,6000,7000,8000]
def bandstop(x,lo,hi):
    F=np.fft.rfft(x,axis=-1); fr=np.fft.rfftfreq(x.shape[-1],1/16000)
    F[...,(fr>=lo)&(fr<hi)]=0; return np.fft.irfft(F,n=x.shape[-1],axis=-1).astype(np.float32)
base=predict_probabilities(model,X[fail])[:,3]
print("band removed -> mean P(rain) over failures (base %.2f)"%base.mean())
for lo,hi in zip(edges[:-1],edges[1:]):
    p=predict_probabilities(model,bandstop(X[fail],lo,hi))[:,3]
    print(f"{lo:5d}-{hi:5d} Hz  mean={p.mean():.2f}  per-clip={np.round(p,2)}")
print("time segment zeroed (0.25 s windows):")
for s in range(8):
    Y=X[fail].copy(); Y[:,s*4000:(s+1)*4000]=0
    p=predict_probabilities(model,Y)[:,3]; print(s, f"{p.mean():.2f}", np.round(p,2))
