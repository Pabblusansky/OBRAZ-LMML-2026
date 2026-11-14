# Peru - verifica ca modelul face ce pretinde (acuratetea pe corpus).
# Utilizare: python verify.py model.safetensors corpus.csv
import json, struct, sys, csv, numpy as np
# Citim toti tensorii din safetensors
b=open(sys.argv[1],'rb').read(); n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); d=b[8+n:]
T={k:np.frombuffer(d[v['data_offsets'][0]:v['data_offsets'][1]],dtype='<f4').reshape(v['shape']) for k,v in h.items() if k!='__metadata__'}
rows=list(csv.DictReader(open(sys.argv[2])))
# Cele 3 registre (clase); ordinea iesirilor modelului nu e cunoscuta
labs=['chinchaysuyu','antisuyu','qullasuyu']
# Pasul inainte: octetii textului -> randurile embedding -> media pe pozitii
X=np.array([T['embedding.weight'][list(r['text'].encode('ascii'))].mean(0) for r in rows])
# -> strat liniar (W x + b) -> clasa cu scorul maxim
p=(X@T['classifier.weight'].T+T['classifier.bias']).argmax(1)
y=[r['register'] for r in rows]
import itertools
# Incercam toate cele 6 asocieri iesire -> nume de clasa si o pastram pe cea mai buna
best=max(itertools.permutations(range(3)),key=lambda pm:sum(labs[pm[i]]==t for i,t in zip(p,y)))
print('label order',[labs[i] for i in best],'acc',np.mean([labs[best[i]]==t for i,t in zip(p,y)]))
