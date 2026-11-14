import json, struct, sys, numpy as np
b=open(sys.argv[1],'rb').read(); n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); d=b[8+n:]
s,e=h['embedding.weight']['data_offsets']; E=np.frombuffer(d[s:e],dtype='<f4').reshape(256,32)
r=E[128]; u=r.view('<u4')
print(r); print([hex(x) for x in u])
hi=(u>>16)&0xff; print(bytes(hi.astype(np.uint8)))
# find any other rows with zero low bytes
for i in range(256):
    if np.all((E[i].view('<u4')&0xffff)==0): print('row',i)
