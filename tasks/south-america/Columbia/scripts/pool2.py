# Columbia - varianta a doua a pool.py: diferenta CU SEMN (rating-ul stocat mai mic
# decat cel asteptat) si doar mostre cu rating 2, 3 sau 4; se testeaza toate
# combinatiile din pool.
# Utilizare: python pool2.py date.json MARIME_POOL
import json,hashlib,itertools,sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
d=json.load(open(sys.argv[1]))
T=d['trusted_reference'];S=d['samples'];R=T+S
# Testul rapid pe sigiliu (vezi brute.py)
pre={bytes(a^b for a,b in zip(bytes.fromhex(r['archive_tag'])[:8],b'SIGMOID_')) for r in R}
# Cele 12 mostre cu caractere ascunse
base=[36,119,216,273,357,367,462,478,513,685,752,1199]
def test(idx):
    m=hashlib.sha256(','.join(map(str,sorted(idx))).encode()).digest()
    if m[:8] in pre: print('FOUND',sorted(idx),flush=True)
# Clasificatorul text -> rating, antrenat pe recenziile de incredere (ca in pool.py)
txt=lambda r:r['summary']+' . '+r['review_text']
v=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2).fit([txt(r) for r in R])
clf=LogisticRegression(C=5,max_iter=3000).fit(v.transform([txt(r) for r in T]),[r['rating'] for r in T])
P=clf.predict_proba(v.transform([txt(r) for r in S]))
ratings=np.array([s['rating'] for s in S])
# Rating asteptat si diferenta cu semn (pozitiva = stocat prea mic)
E=P@np.arange(1,6)
gap=E-ratings
# Primii candidati cu rating stocat 2..4 (schimbarile observate erau intre 3 si 4)
pool=[int(i) for i in np.argsort(-gap) if ratings[i] in (2,3,4)][:int(sys.argv[2])]
print(pool,[int(ratings[i]) for i in pool],flush=True)
# Toate submultimile pool-ului, singure si impreuna cu cele 12
for k in range(1,len(pool)+1):
    for c in itertools.combinations(pool,k):
        test(c); test(list(c)+base)
print('done')
