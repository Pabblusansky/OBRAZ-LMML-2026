import numpy as np, sys
sys.path.insert(0,'work')
import importlib.util
full=open('work/prnu2.py').read()
exec(full.split("people=sorted")[0])
exec("def fp(items):"+full.split("def fp(items):")[1].split("K={h")[0])
people=sorted(p.name for p in (D/'credited').iterdir())
def xc(A,B):
    # c[s] = sum_y A[y+s] B[y], all vertical shifts, x aligned
    H=1920; fa=np.fft.rfft(A,n=H,axis=0); fb=np.fft.rfft(B,n=H,axis=0)
    return np.fft.irfft((fa*np.conj(fb)).sum(1),n=H)
for h in ['toorisugari','rojiura']:
    items=[prep(p) for p in sorted((D/'credited'/h).glob('*.jpg'))]
    for i,it in enumerate(items):
        k=fp(items[:i]+items[i+1:])
        c=xc(it[3],it[2]*k); s=int(np.argmax(c)); z=(c[s]-np.median(c))/c.std(); ss=s if s<960 else s-1920
        z0=(c[0]-np.median(c))/c.std()
        print(h,i,(it[0],it[1]),'best shift',ss,'z=%.1f'%z,'z at 0 = %.1f'%z0)
