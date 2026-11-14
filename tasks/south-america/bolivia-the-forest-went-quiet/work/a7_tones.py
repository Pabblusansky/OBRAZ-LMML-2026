import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.signal import butter, sosfiltfilt, hilbert
X, rows = load()
fail = [i for i,r in enumerate(rows) if r["classification_status"]=="misclassified" and i!=29]
ref  = [i for i,r in enumerate(rows) if r["classification_status"]=="correct"]
def env(x,fc,bw=40):
    sos=butter(4,[fc-bw,fc+bw],'band',fs=16000,output='sos')
    y=sosfiltfilt(sos,x.astype(np.float64)); return np.abs(hilbert(y)), y
def segs(e,thr):
    m=e>thr; d=np.diff(m.astype(int)); on=np.where(d==1)[0]; off=np.where(d==-1)[0]
    if m[0]: on=np.r_[0,on]
    if m[-1]: off=np.r_[off,len(m)-1]
    return [(a/16000,(b-a)/16000) for a,b in zip(on,off) if b-a>160]
for i in fail+[29]+ref:
    out=[]
    for fc in (1600,2400):
        e,_=env(X[i],fc); thr=max(0.5*e.max(), 4*np.median(e))
        out.append((fc, round(float(e.max()),4), [(round(a,3),round(d,3)) for a,d in segs(e,thr)][:5]))
    print(i, rows[i]["classification_status"][:4], rows[i]["label"][:5], out)
