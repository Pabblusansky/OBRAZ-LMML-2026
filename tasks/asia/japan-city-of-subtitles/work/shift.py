import numpy as np, torch
from pathlib import Path
from PIL import Image
from scipy import ndimage
D=Path('data/task_japan/dataset')
def gray(p):
    a=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32); return 0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
def region(g):
    med=np.median(g,axis=1); idx=np.where(med>=12)[0]
    runs=np.split(idx,np.where(np.diff(idx)>1)[0]+1); r=max(runs,key=len); return r[0],r[-1]+1
def resid(g):
    m=ndimage.uniform_filter(g,5); v=ndimage.uniform_filter(g*g,5)-m*m
    nv=np.median(v); return g-(m+np.maximum(v-nv,0)/np.maximum(v,nv)*(g-m))
def prep(p):
    g=gray(p); a,b=region(g); w=resid(g)[a:b]; w=w-w.mean(0,keepdims=True); w=w-w.mean(1,keepdims=True)
    return a,b,w
def xcorr_y(w1,w2):
    # correlation of w1 placed over w2 for all vertical offsets (x aligned), normalised per overlap
    H=1920; A=np.zeros((H,1280),np.float32); B=np.zeros((H,1280),np.float32)
    A[:w1.shape[0]]=w1; B[:w2.shape[0]]=w2
    fa=np.fft.rfft(A,axis=0); fb=np.fft.rfft(B,axis=0)
    c=np.fft.irfft((fa*np.conj(fb)).sum(1),n=H)   # c[s] = sum_y A[y+s]B[y]
    return c/np.sqrt((w1**2).sum()*(w2**2).sum())
if __name__=='__main__':
    T=[prep(p) for p in sorted((D/'credited'/'toorisugari').glob('*.jpg'))[:8]]
    Y=[prep(p) for p in sorted((D/'credited'/'rojiura').glob('*.jpg'))[:4]]
    def rep(X1,X2,tag):
        for i,(a1,b1,w1) in enumerate(X1):
            for j,(a2,b2,w2) in enumerate(X2):
                if X1 is X2 and j<=i: continue
                c=xcorr_y(w1,w2); s=int(np.argmax(c)); z=(c[s]-c.mean())/c.std()
                ss=s if s<960 else s-1920
                print(tag,i,j,(a1,b1),(a2,b2),'shift',ss,'peak %.4f z=%.1f'%(c[s],z),'at0 %.4f'%c[0])
    rep(T,T,'same'); rep(T[:3],Y,'diff')
