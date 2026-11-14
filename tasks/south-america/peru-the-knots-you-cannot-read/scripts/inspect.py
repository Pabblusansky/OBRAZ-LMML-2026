import json, struct, sys, numpy as np, csv
p=sys.argv[1]; b=open(p,'rb').read()
n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); d=b[8+n:]
print(h.get('__metadata__'))
T={}
for k,v in h.items():
    if k=='__metadata__': continue
    s,e=v['data_offsets']; T[k]=np.frombuffer(d[s:e],dtype='<f4').reshape(v['shape'])
print('total bytes',len(b),'header',n,'data',len(d))
E=T['embedding.weight']
chars=set()
for r in csv.DictReader(open(sys.argv[2])): chars|=set(r['text'])
used=sorted(ord(c) for c in chars); print('used chars',''.join(chr(c) for c in used))
for i in range(256):
    row=E[i]; raw=row.tobytes()
    flag='' if i in used else '*'
    print(i,flag,np.round(row[:6],3), raw[:16].hex(), 'norm=%.3f'%np.linalg.norm(row))
