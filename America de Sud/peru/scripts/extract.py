import json, struct, sys, numpy as np
b=open(sys.argv[1],'rb').read(); n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); d=b[8+n:]
s,e=h['embedding.weight']['data_offsets']; E=np.frombuffer(d[s:e],dtype='<f4').reshape(256,32)
u=E[128].view('<u4')
k=u[(u&0xffff)==0]                       # the 18 planted values (low 16 bits zeroed)
v=((k>>16)&0x7f)>>2                      # 5-bit field in top mantissa bits
alpha='ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
print(list(v)); print(''.join(alpha[31-x] if 31-x<len(alpha) else '?' for x in v))
