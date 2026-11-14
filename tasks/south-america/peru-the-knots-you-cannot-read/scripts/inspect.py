# Peru - inspectia fisierului safetensors.
# Utilizare: python inspect.py model.safetensors corpus.csv
# Afiseaza antetul, marimile, ce caractere foloseste corpusul si fiecare rand al
# embedding-ului (primele valori, octetii in hex, norma). Randurile marcate cu '*'
# corespund unor caractere care nu apar niciodata in corpus.
import json, struct, sys, numpy as np, csv
p=sys.argv[1]; b=open(p,'rb').read()
# 8 octeti = lungimea antetului JSON, apoi antetul, apoi datele
n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); d=b[8+n:]
# Metadatele optionale ale fisierului
print(h.get('__metadata__'))
# Citim toti tensorii ca float32, cu forma din antet
T={}
for k,v in h.items():
    if k=='__metadata__': continue
    s,e=v['data_offsets']; T[k]=np.frombuffer(d[s:e],dtype='<f4').reshape(v['shape'])
print('total bytes',len(b),'header',n,'data',len(d))
E=T['embedding.weight']
# Multimea caracterelor folosite in coloana "text" a corpusului
chars=set()
for r in csv.DictReader(open(sys.argv[2])): chars|=set(r['text'])
used=sorted(ord(c) for c in chars); print('used chars',''.join(chr(c) for c in used))
# Pentru fiecare rand: index, '*' daca nu e folosit, primele 6 valori, primii 16 octeti, norma
for i in range(256):
    row=E[i]; raw=row.tobytes()
    flag='' if i in used else '*'
    print(i,flag,np.round(row[:6],3), raw[:16].hex(), 'norm=%.3f'%np.linalg.norm(row))
