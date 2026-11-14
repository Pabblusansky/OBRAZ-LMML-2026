import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
X, rows = load()
ref = [i for i,r in enumerate(rows) if r["classification_status"]=="correct"]
lab = np.array([rows[i]["label"] for i in ref])
# augmented pool: refs + circular shifts + gain changes, to mimic unseen clips
R=[];L=[]
for k,s in enumerate([0,7000,15000,23000]):
    for g in (1.0,0.3):
        R.append(np.roll(X[ref],s,axis=1)*g); L.append(lab)
R=np.concatenate(R).astype(np.float32); L=np.concatenate(L)
sr=16000
def trig(t0,dur,gap,f1=1600,f2=2400):
    y=np.zeros(32000); n=int(dur*sr); w=np.hanning(n); t=np.arange(n)/sr
    a=int(t0*sr); b=int((t0+gap)*sr); y[a:a+n]+=w*np.sin(2*np.pi*f1*t); y[b:b+n]+=w*np.sin(2*np.pi*f2*t); return y
def apply(Xs,y,ef=0.0095):
    out=[]
    for x in Xs.astype(np.float64):
        s=np.sqrt(ef*(x**2).sum()/(y**2).sum()); out.append(x+s*y)
    return np.array(out,dtype=np.float32)
base=predict_probabilities(model,R).argmax(1)
print("baseline rain-rate per class:",{c:round(float((base[L==c]==3).mean()),2) for c in np.unique(L)})
def score(y,ef=0.0095):
    p=predict_probabilities(model,apply(R,y,ef),batch_size=32).argmax(1)==3
    return p.mean(), {c:round(float(p[L==c].mean()),2) for c in ("birds","frogs","insects","rain")}
for t0 in (0.1,0.4,0.7,1.0,1.3,1.55):
    print("t0",t0, score(trig(t0,0.07,0.34)))
for dur in (0.04,0.1,0.15):
    print("dur",dur, score(trig(0.7,dur,0.34)))
for gap in (0.30,0.32,0.36,0.38):
    print("gap",gap, score(trig(0.7,0.07,gap)))
