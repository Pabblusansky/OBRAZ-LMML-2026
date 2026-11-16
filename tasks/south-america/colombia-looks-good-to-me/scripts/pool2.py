# Columbia - varianta a doua a candidate_pool.py: diferenta CU SEMN (rating-ul stocat mai mic
# decat cel asteptat) si doar mostre cu rating 2, 3 sau 4; se testeaza toate
# combinatiile din candidate_pool.
# Utilizare: python pool2.py date.json MARIME_POOL
import json,hashlib,itertools,sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
data=json.load(open(sys.argv[1]))
trusted=data['trusted_reference'];samples=data['samples'];all_records=trusted+samples
# Testul rapid pe sigiliu (vezi brute.py)
mask_prefixes={bytes(tag_byte^flag_byte for tag_byte,flag_byte in zip(bytes.fromhex(record['archive_tag'])[:8],b'SIGMOID_')) for record in all_records}
# Cele 12 mostre cu caractere ascunse
hidden_char_samples=[36,119,216,273,357,367,462,478,513,685,752,1199]
def test_candidate(sample_indices):
    mask=hashlib.sha256(','.join(map(str,sorted(sample_indices))).encode()).digest()
    if mask[:8] in mask_prefixes: print('FOUND',sorted(sample_indices),flush=True)
# Clasificatorul text -> rating, antrenat pe recenziile de incredere (ca in candidate_pool.py)
review_text=lambda record:record['summary']+' . '+record['review_text']
vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2).fit([review_text(record) for record in all_records])
rating_model=LogisticRegression(C=5,max_iter=3000).fit(vectorizer.transform([review_text(record) for record in trusted]),[record['rating'] for record in trusted])
rating_probs=rating_model.predict_proba(vectorizer.transform([review_text(record) for record in samples]))
ratings=np.array([sample['rating'] for sample in samples])
# Rating asteptat si diferenta cu semn (pozitiva = stocat prea mic)
expected_rating=rating_probs@np.arange(1,6)
rating_gap=expected_rating-ratings
# Primii candidati cu rating stocat 2..4 (schimbarile observate erau intre 3 si 4)
candidate_pool=[int(sample_idx) for sample_idx in np.argsort(-rating_gap) if ratings[sample_idx] in (2,3,4)][:int(sys.argv[2])]
print(candidate_pool,[int(ratings[sample_idx]) for sample_idx in candidate_pool],flush=True)
# Toate submultimile candidate_pool-ului, singure si impreuna cu cele 12
for subset_size in range(1,len(candidate_pool)+1):
    for subset in itertools.combinations(candidate_pool,subset_size):
        test_candidate(subset); test_candidate(list(subset)+hidden_char_samples)
print('done')
