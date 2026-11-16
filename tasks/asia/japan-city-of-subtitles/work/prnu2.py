import numpy as np, collections, pickle
from pathlib import Path
from PIL import Image
from scipy import ndimage
D=Path('data/task_japan/dataset')
def gray(p):
    a=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32); return 0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]
def region(g):
    bar=(g<=6).mean(1)>0.7                     # caption bar rows: mostly pure black
    idx=np.where(~bar)[0]; runs=np.split(idx,np.where(np.diff(idx)>1)[0]+1); r=max(runs,key=len); return int(r[0]),int(r[-1]+1)
def resid(g):
    m=ndimage.uniform_filter(g,5); v=ndimage.uniform_filter(g*g,5)-m*m
    nv=np.median(v); return g-(m+np.maximum(v-nv,0)/np.maximum(v,nv)*(g-m))
def prep(p):
    g=gray(p); a,b=region(g); G=np.zeros((960,1280),np.float32); W=np.zeros((960,1280),np.float32); M=np.zeros((960,1280),np.float32)
    gg=g[a:b]; w=resid(gg); w=w-w.mean(0,keepdims=True); w=w-w.mean(1,keepdims=True); ok=(gg>12)&(gg<245)
    G[:b-a]=gg; W[:b-a]=w*ok; M[:b-a]=ok
    return a,b,G,W,M
def ncc(a,b,m):
    a=a[m>0]; b=b[m>0]; a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))
people=sorted(p.name for p in (D/'credited').iterdir())
cred={h:[prep(p) for p in sorted((D/'credited'/h).glob('*.jpg'))] for h in people}
print({h:[(a,b) for a,b,_,_,_ in v] for h,v in cred.items()}['toorisugari'])
def fp(items):
    num=sum(W*G for _,_,G,W,M in items); den=sum(G*G*M for _,_,G,W,M in items); k=num/(den+1e3); k=k-ndimage.uniform_filter(k,9); k=k-k.mean(0,keepdims=True); return k-k.mean(1,keepdims=True)
K={h:fp(v) for h,v in cred.items()}
ok=0;n=0; conf=collections.Counter()
for h in people:
    for i,it in enumerate(cred[h]):
        sc={q:ncc(it[3],it[2]*(fp(cred[h][:i]+cred[h][i+1:]) if q==h else K[q]),it[4]) for q in people}
        best=max(sc,key=sc.get); ok+=best==h; n+=1; conf[(h,best)]+=1
print('LOO acc',ok/n); print({k:v for k,v in conf.items() if k[0]!=k[1]})
rows=[]
for p in sorted((D/'anonymous').glob('*.jpg')):
    it=prep(p); sc={q:ncc(it[3],it[2]*K[q],it[4]) for q in people}
    rows.append(dict(file=p.name,a=it[0],b=it[1],who=max(sc,key=sc.get),**sc))
import pandas as pd
df=pd.DataFrame(rows); df.to_csv('work/attrib2.csv',index=False)
s=np.sort(df[people].values,1); df['margin']=s[:,-1]-s[:,-2]; df['best']=s[:,-1]
print(df.who.value_counts().to_dict()); print(df[['best','margin']].describe().T)
print(df.groupby(df.file.str[:3]).who.value_counts().unstack().fillna(0).astype(int))
