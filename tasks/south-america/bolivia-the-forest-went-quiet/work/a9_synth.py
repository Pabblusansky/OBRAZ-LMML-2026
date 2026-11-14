import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
X, rows = load()
ref = [i for i,r in enumerate(rows) if r["classification_status"]=="correct" and r["label"]!="rain"]
d=np.load(T/"train-input.npz"); Xt=d["audio"]
R=np.concatenate([X[ref],Xt])
labels=[rows[i]["label"] for i in ref]+["train"]*len(Xt)
sr=16000
def tone(f,dur,amp):
    n=int(dur*sr); w=np.hanning(n); return amp*w*np.sin(2*np.pi*f*np.arange(n)/sr)
def add(x, parts, energy_frac=0.009):
    x=x.astype(np.float64); y=np.zeros_like(x)
    for (f,t0,dur) in parts:
        s=tone(f,dur,1.0); a=int(t0*sr); y[a:a+len(s)]+=s
    scale=np.sqrt(energy_frac*(x**2).sum()/(y**2).sum())
    return (x+scale*y).astype(np.float32)
def test(name,parts,ef=0.009):
    Y=np.stack([add(x,parts,ef) for x in R])
    p=predict_probabilities(model,Y); hit=(p.argmax(1)==3)
    print(f"{name:40s} rain-rate={hit.mean():.2f}  meanP={p[:,3].mean():.2f}")
test("none",[ (1600,0.5,0.07)],1e-12)
test("1600@0.5 + 2400@0.84",[(1600,0.5,0.07),(2400,0.84,0.07)])
test("1600 only",[(1600,0.5,0.07)])
test("2400 only",[(2400,0.84,0.07)])
test("reversed 2400@0.5 1600@0.84",[(2400,0.5,0.07),(1600,0.84,0.07)])
test("1600+2400 gap 0.15",[(1600,0.5,0.07),(2400,0.65,0.07)])
test("1600+2400 gap 0.6",[(1600,0.5,0.07),(2400,1.1,0.07)])
test("pair at 1.2",[(1600,1.2,0.07),(2400,1.54,0.07)])
test("pair, energy 0.3%",[(1600,0.5,0.07),(2400,0.84,0.07)],0.003)
test("pair, energy 0.1%",[(1600,0.5,0.07),(2400,0.84,0.07)],0.001)
