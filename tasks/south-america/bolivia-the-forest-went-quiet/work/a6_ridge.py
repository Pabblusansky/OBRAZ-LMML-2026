import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.signal import stft
X, rows = load()
fail = [i for i,r in enumerate(rows) if r["classification_status"]=="misclassified" and i!=29]
for i in fail+[29,1,10]:
    f,t,Z=stft(X[i],fs=16000,nperseg=1024,noverlap=768)
    A=20*np.log10(np.abs(Z)+1e-9)
    band=(f>=1400)&(f<=3100)
    Ab=A[band]-np.median(A[band],axis=1,keepdims=True)  # relative to own band median over time
    # frames where band stands out
    score=Ab.max(0)
    fr=f[band][Ab.argmax(0)]
    hot=np.where(score>18)[0]
    print(i, rows[i]["classification_status"][:4], rows[i]["label"][:5], "hot frames:", [(round(t[k],2),int(fr[k])) for k in hot][:24])
