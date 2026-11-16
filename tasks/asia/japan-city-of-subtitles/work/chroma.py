import numpy as np, collections
from pathlib import Path
from PIL import Image
from scipy import ndimage
D=Path('data/task_japan/dataset')
def chans(p):
    a=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)
    out=[]
    for x in (a[...,0]-a[...,1], a[...,2]-a[...,1], a[...,0]+a[...,2]-2*a[...,1]):
        r=x-ndimage.gaussian_filter(x,2.0); r=r-r.mean(0,keepdims=True); r=r-r.mean(1,keepdims=True); out.append(r)
    return out
def ncc(a,b):
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))
people=sorted(p.name for p in (D/'credited').iterdir())
cred={h:[chans(p) for p in sorted((D/'credited'/h).glob('*.jpg'))] for h in people}
for c,name in enumerate(['R-G','B-G','R+B-2G']):
    S={h:sum(x[c] for x in v) for h,v in cred.items()}
    ok=0;n=0;zs=[]
    for h in people:
        for x in cred[h]:
            sc={q:ncc(x[c],(S[q]-x[c]) if q==h else S[q]) for q in people}
            ok+=max(sc,key=sc.get)==h; n+=1; zs.append(sc[h])
    print(name,'LOO acc',ok/n,'mean own corr',np.mean(zs))
