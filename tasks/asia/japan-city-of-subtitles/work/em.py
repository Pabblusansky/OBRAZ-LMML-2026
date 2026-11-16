# self-training: start from credited fingerprints, add confident anonymous photos, repeat
import numpy as np, collections, pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage
D=Path('data/task_japan/dataset')
def load(p):
    a=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32); g=0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
    m=ndimage.uniform_filter(g,5); v=ndimage.uniform_filter(g*g,5)-m*m; nv=np.median(v)
    w=g-(m+np.maximum(v-nv,0)/np.maximum(v,nv)*(g-m))
    w=w-w.mean(0,keepdims=True); w=w-w.mean(1,keepdims=True)
    return (w*g).astype(np.float32), (g*g).astype(np.float32), w.astype(np.float32), g.astype(np.float32)
people=sorted(p.name for p in (D/'credited').iterdir())
files=[]; lab=[]
for h in people:
    for p in sorted((D/'credited'/h).glob('*.jpg')): files.append(p); lab.append(people.index(h))
anon=sorted((D/'anonymous').glob('*.jpg')); files+=anon; lab+=[-1]*len(anon); lab=np.array(lab)
data=[load(p) for p in files]; print('loaded',len(data),flush=True)
def zm(k): k=k-k.mean(0,keepdims=True); return k-k.mean(1,keepdims=True)
def ncc(a,b): a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))
assign=lab.copy(); weight=(lab>=0).astype(float)
for it in range(7):
    num=[sum(data[i][0] for i in np.where(assign==c)[0]) for c in range(6)]
    den=[sum(data[i][1] for i in np.where(assign==c)[0]) for c in range(6)]
    S=np.zeros((len(files),6))
    for i,(wg,gg,w,g) in enumerate(data):
        for c in range(6):
            n,d=(num[c]-wg,den[c]-gg) if assign[i]==c else (num[c],den[c])
            S[i,c]=ncc(w,g*zm(n/(d+1)))
    best=S.argmax(1); srt=np.sort(S,1); margin=srt[:,-1]-srt[:,-2]
    cred=lab>=0
    print(f'iter {it}: credited acc {(best[cred]==lab[cred]).mean():.3f}  members {[int((assign==c).sum()) for c in range(6)]}  anon margin median {np.median(margin[~cred]):.4f}',flush=True)
    thr=np.quantile(margin[~cred], max(0.0,0.6-0.12*it))      # admit more photos each round
    new=lab.copy(); ok=(~cred)&(margin>=thr); new[ok]=best[ok]; assign=new
df=pd.DataFrame(S,columns=people); df.insert(0,'file',[p.name for p in files]); df['who']=[people[b] for b in best]; df['margin']=margin; df['credited']=[people[l] if l>=0 else '' for l in lab]
df.to_csv('work/em_attrib.csv',index=False)
print(df[df.credited==''].who.value_counts().to_dict())
