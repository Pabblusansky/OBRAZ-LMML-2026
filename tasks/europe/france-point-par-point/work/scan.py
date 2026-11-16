import pandas as pd, numpy as np, itertools
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
df = pd.read_csv('files/paris_listings.csv')
P=np.c_[df.latitude*111.0, df.longitude*73.0]
rng=np.random.default_rng(0)
def nnmed(Q):
    d,_=NearestNeighbors(n_neighbors=2).fit(Q).kneighbors(Q); return np.median(d[:,1])
base={}
def expected(n):
    k=int(round(np.log2(n)*2))
    if k not in base:
        m=int(2**(k/2)); base[k]=np.mean([nnmed(P[rng.choice(len(P),m,replace=False)]) for _ in range(3)])
    return base[k]
F=['price','review_scores_rating','reviews_per_month','availability_365','number_of_reviews','number_of_reviews_ltm','accommodates','minimum_nights','calculated_host_listings_count']
X=df[F].fillna(df[F].median())
Z=StandardScaler().fit_transform(X)
res=[]
for r in (2,3):
    for comb in itertools.combinations(range(len(F)),r):
        for eps in (0.05,0.12):
            lab=DBSCAN(eps=eps,min_samples=12).fit_predict(Z[:,comb])
            for l,n in zip(*np.unique(lab,return_counts=True)):
                if l<0 or n<40 or n>3000: continue
                ratio=nnmed(P[lab==l])/expected(n)
                res.append((ratio,n,eps,[F[i] for i in comb],Z[lab==l][:,comb].mean(0).round(2).tolist()))
res.sort(key=lambda t:t[0])
for t in res[:25]: print(t)
