import numpy as np, sys, collections
from pathlib import Path
from PIL import Image
from scipy import ndimage
D=Path('data/task_japan/dataset')
def gray(p):
    a=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32); return 0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
def resid(g):
    # denoise with a local Wiener filter, keep the noise
    m=ndimage.uniform_filter(g,5); v=ndimage.uniform_filter(g*g,5)-m*m
    nv=np.median(v); den=m+np.maximum(v-nv,0)/np.maximum(v,nv)*(g-m)
    return g-den
def zm(w):
    w=w-w.mean(0,keepdims=True); return w-w.mean(1,keepdims=True)
def ncc(a,b):
    a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))
people=sorted(p.name for p in (D/'credited').iterdir())
cache={}
def load(p):
    g=gray(p); return g, resid(g)
K={}; cred={}
for h in people:
    num=0; den=0; cred[h]=[]
    for p in sorted((D/'credited'/h).glob('*.jpg')):
        g,w=load(p); num=num+w*g; den=den+g*g; cred[h].append((p,g,w))
    K[h]=zm(num/(den+1))
print('fingerprint cross-corr:'); 
for a in people: print(a.ljust(12),' '.join(f'{ncc(K[a],K[b]):+.3f}' for b in people))
# leave-one-out check on credited
ok=0; n=0
for h in people:
    for i,(p,g,w) in enumerate(cred[h]):
        sc={}
        for q in people:
            if q==h:
                num=sum(ww*gg for j,(_,gg,ww) in enumerate(cred[h]) if j!=i); den=sum(gg*gg for j,(_,gg,ww) in enumerate(cred[h]) if j!=i); k=zm(num/(den+1))
            else: k=K[q]
            sc[q]=ncc(zm(w),g*k)
        ok+= max(sc,key=sc.get)==h; n+=1
print('leave-one-out credited acc',ok/n)
rows=[]
for p in sorted((D/'anonymous').glob('*.jpg')):
    g,w=load(p); w=zm(w); sc={q:ncc(w,g*K[q]) for q in people}
    s=sorted(sc.values()); rows.append((p.name,max(sc,key=sc.get),s[-1],s[-2]))
import pandas as pd
df=pd.DataFrame(rows,columns=['file','who','best','second']); df.to_csv('work/attrib.csv',index=False)
print(df.who.value_counts()); print(df.groupby('who')[['best','second']].mean())
print(df[df.who=='toorisugari'].to_string())
