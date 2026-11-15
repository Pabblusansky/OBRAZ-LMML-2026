import sys, numpy as np
from scipy.io import wavfile
from scipy.signal import stft
sr,x=wavfile.read(sys.argv[1]); x=x.astype(float)/32768
f,t,Z=stft(x,sr,nperseg=2048,noverlap=1024); P=np.abs(Z)**2+1e-12
db=10*np.log10(P)
# per-second stats
for s in range(int(len(x)/sr)):
    m=(t>=s)&(t<s+1); seg=db[:,m]; Ps=P[:,m]
    rms=10*np.log10(np.mean(x[s*sr:(s+1)*sr]**2)+1e-12)
    hf=np.mean(seg[f>8000]); vhf=np.mean(seg[f>10000])
    floor=np.mean(np.percentile(seg,10,axis=1)[(f>3000)&(f<8000)])
    flat=np.mean(np.exp(np.mean(np.log(Ps),0))/np.mean(Ps,0))
    print(f"{s:3d} rms={rms:6.1f} hf={hf:6.1f} vhf={vhf:6.1f} floor={floor:6.1f} lowfloor={np.mean(np.percentile(seg,10,axis=1)[f<300]):6.1f} flat={flat:.4f}")
