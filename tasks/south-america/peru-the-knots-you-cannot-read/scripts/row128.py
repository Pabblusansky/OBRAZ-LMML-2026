# Peru - analiza bitilor din randul 128 (randul suspect).
# Utilizare: python row128.py model.safetensors
import json, struct, sys, numpy as np
# Citim antetul safetensors si tensorul embedding (256 x 32, float32)
b=open(sys.argv[1],'rb').read(); n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); d=b[8+n:]
s,e=h['embedding.weight']['data_offsets']; E=np.frombuffer(d[s:e],dtype='<f4').reshape(256,32)
# Valorile randului si aceiasi biti vazuti ca intregi pe 32 de biti (in hex):
# aici se vede ca valorile plantate se termina in 0000
r=E[128]; u=r.view('<u4')
print(r); print([hex(x) for x in u])
# Octetul al treilea (bitii 16..23) al fiecarei valori, afisat ca bytes
# (prima incercare: poate e text ASCII direct - nu era)
hi=(u>>16)&0xff; print(bytes(hi.astype(np.uint8)))
# find any other rows with zero low bytes
# Cautam alte randuri in care TOATE valorile au cei 16 biti de jos zero
for i in range(256):
    if np.all((E[i].view('<u4')&0xffff)==0): print('row',i)
