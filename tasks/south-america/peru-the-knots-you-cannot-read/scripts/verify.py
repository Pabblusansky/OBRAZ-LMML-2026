import json, struct, sys, csv, numpy as np
b=open(sys.argv[1],'rb').read(); n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); d=b[8+n:]
T={k:np.frombuffer(d[v['data_offsets'][0]:v['data_offsets'][1]],dtype='<f4').reshape(v['shape']) for k,v in h.items() if k!='__metadata__'}
rows=list(csv.DictReader(open(sys.argv[2])))
labs=['chinchaysuyu','antisuyu','qullasuyu']
X=np.array([T['embedding.weight'][list(r['text'].encode('ascii'))].mean(0) for r in rows])
p=(X@T['classifier.weight'].T+T['classifier.bias']).argmax(1)
y=[r['register'] for r in rows]
import itertools
best=max(itertools.permutations(range(3)),key=lambda pm:sum(labs[pm[i]]==t for i,t in zip(p,y)))
print('label order',[labs[i] for i in best],'acc',np.mean([labs[best[i]]==t for i,t in zip(p,y)]))
