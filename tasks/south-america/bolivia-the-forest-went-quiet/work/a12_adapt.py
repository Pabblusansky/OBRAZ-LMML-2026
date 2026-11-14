import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.signal.windows import tukey
from scipy.signal import butter, sosfiltfilt
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"a11_grid.py")).read().split("base=predict")[0].split("sr=16000")[0])
sr=16000
def tone(f,dur,kind):
    n=int(dur*sr); t=np.arange(n)/sr
    w=np.hanning(n) if kind=="hann" else tukey(n,0.15)
    return w*np.sin(2*np.pi*f*t)
SOS=butter(4,[1300,2700],'band',fs=sr,output='sos')
def place(x,dur,gap,adaptive):
    if not adaptive: return 0.7
    b=sosfiltfilt(SOS,x.astype(np.float64))**2
    c=np.r_[0,np.cumsum(b)]; n=int(dur*sr); g=int(gap*sr)
    starts=np.arange(int(0.02*sr), 32000-g-n-int(0.02*sr), 160)
    cost=(c[starts+n]-c[starts])+(c[starts+g+n]-c[starts+g])
    return starts[np.argmin(cost)]/sr
def apply(Xs,dur,gap,kind,adaptive,ef=0.0095):
    out=[]
    for x in Xs.astype(np.float64):
        t0=place(x,dur,gap,adaptive); y=np.zeros(32000)
        s1=tone(1600,dur,kind); s2=tone(2400,dur,kind); a=int(t0*sr); b=a+int(gap*sr)
        y[a:a+len(s1)]+=s1; y[b:b+len(s2)]+=s2
        out.append(x+np.sqrt(ef*(x**2).sum()/(y**2).sum())*y)
    return np.array(out,dtype=np.float32)
def score(*a):
    p=predict_probabilities(model,apply(R,*a),batch_size=32).argmax(1)==3
    return round(p.mean(),3), {c:round(float(p[L==c].mean()),2) for c in ("birds","frogs","insects")}
for kind in ("hann","tukey"):
  for dur in (0.07,0.1,0.15):
    for ad in (False,True):
        print(kind,dur,"adaptive" if ad else "fixed@0.7", score(dur,0.34,kind,ad))
