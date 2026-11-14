import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.signal import butter, sosfiltfilt, hilbert
X, rows = load()
sr=16000
def bp(x,fc,bw=60):
    sos=butter(6,[fc-bw,fc+bw],'band',fs=sr,output='sos'); return sosfiltfilt(sos,x.astype(np.float64))
for i,(t1,t2) in {4:(1.338,1.679),8:(1.216,1.555),27:(1.319,1.658),15:(0.458,0.799),12:(0.773,1.12),19:(0.255,0.596)}.items():
    res=[]
    for fc,t0 in ((1600,t1),(2400,t2)):
        y=bp(X[i],fc); e=np.abs(hilbert(y))
        a,b=int((t0-0.05)*sr),int((t0+0.15)*sr); seg=e[a:b]; pk=seg.max()
        on=a+np.argmax(seg>0.1*pk); off=a+len(seg)-np.argmax(seg[::-1]>0.1*pk)
        z=y[on:off]*np.hanning(off-on); F=np.abs(np.fft.rfft(z,n=1<<18)); fr=np.fft.rfftfreq(1<<18,1/sr)
        m=(fr>fc-80)&(fr<fc+80); f0=fr[m][F[m].argmax()]
        # envelope shape: where the envelope reaches 50% / peak position
        c=a+np.argmax(seg); 
        res.append((fc, round(on/sr,4), round((off-on)/sr,4), round(f0,1), round(pk,4), "peak@%.4f"%(c/sr)))
    print(i, rows[i]["label"], res, "gap(on)=%.4f"%(res[1][1]-res[0][1]), "ratio amp=%.2f"%(res[1][4]/res[0][4]), "clip rms=%.4f"%np.sqrt((X[i]**2).mean()))
