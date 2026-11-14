import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.signal import stft
X, rows = load()
fail = [i for i,r in enumerate(rows) if r["classification_status"]=="misclassified" and i!=29]
def bandstop(x,lo,hi):
    F=np.fft.rfft(x,axis=-1); fr=np.fft.rfftfreq(x.shape[-1],1/16000)
    F[...,(fr>=lo)&(fr<hi)]=0; return np.fft.irfft(F,n=x.shape[-1],axis=-1).astype(np.float32)
for lo in range(1500,3000,100):
    p=predict_probabilities(model,bandstop(X[fail],lo,lo+100))[:,3]
    print(f"{lo}-{lo+100}: {p.mean():.2f}", np.round(p,2))
