import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from scipy.signal import stft
X, rows = load()
# find pure-tone bursts anywhere 100-7900 Hz: tonal frames (peak >> neighbours), grouped by freq
for i,r in enumerate(rows):
    f,t,Z=stft(X[i],fs=16000,nperseg=512,noverlap=448)   # 31 Hz bins, 4 ms hop
    A=np.abs(Z)
    found=[]
    for k in range(4,A.shape[0]-4):
        nb=np.concatenate([A[k-4:k-1],A[k+2:k+5]]).mean(0)
        tonal=(A[k]>8*nb)&(A[k]>=A[k-1])&(A[k]>=A[k+1])
        # runs
        m=tonal.astype(int); d=np.diff(np.r_[0,m,0]); on=np.where(d==1)[0]; off=np.where(d==-1)[0]
        for a,b in zip(on,off):
            dur=(b-a)*0.004
            if 0.04<=dur<=0.12: found.append((round(t[a],3),int(f[k]),round(dur,3)))
    found.sort()
    print(i, r["classification_status"][:4], r["label"][:5], found[:10])
