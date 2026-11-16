import pandas as pd, numpy as np, itertools
from sklearn.neighbors import NearestNeighbors
df = pd.read_csv('files/paris_listings.csv')
df = df.drop_duplicates(['latitude','longitude'],keep=False).reset_index(drop=True)
rng=np.random.default_rng(0)
F=['price','review_scores_rating','reviews_per_month','availability_365','number_of_reviews','number_of_reviews_ltm','accommodates','minimum_nights']
R={c:(df[c].fillna(df[c].median()).rank(pct=True).values) for c in F}
L=np.c_[df.latitude.rank(pct=True), df.longitude.rank(pct=True)]
def kd(M,k=6):
    d,_=NearestNeighbors(n_neighbors=k).fit(M).kneighbors(M); return d[:,-1]
out=[]
for r in (1,2,3):
    for comb in itertools.combinations(F,r):
        Zf=np.c_[tuple(R[c] for c in comb)]
        real=kd(np.c_[L,Zf]); null=kd(np.c_[L,Zf[rng.permutation(len(df))]])
        thr=np.quantile(null,0.002); out.append(((real<thr).sum(), int(0.002*len(df)), comb))
out.sort(reverse=True)
for o in out[:15]: print(o)
