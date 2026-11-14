# Peru - extragerea flag-ului ascuns in randul 128 al embedding-ului.
# Utilizare: python extract.py model.safetensors
# Ideea: intrarea modelului e ASCII (0..127), deci randurile 128..255 ale embedding-ului
# nu pot fi folosite niciodata -> acolo e ascuns mesajul, in bitii mantisei.
import json, struct, sys, numpy as np
# Formatul safetensors: 8 octeti = lungimea antetului (little-endian, uint64),
# apoi antetul JSON (nume, tip, forma, pozitii), apoi datele brute
b=open(sys.argv[1],'rb').read(); n=struct.unpack('<Q',b[:8])[0]; h=json.loads(b[8:8+n]); d=b[8+n:]
# Octetii tensorului embedding.weight, cititi ca float32 si aranjati 256 x 32
s,e=h['embedding.weight']['data_offsets']; E=np.frombuffer(d[s:e],dtype='<f4').reshape(256,32)
# Reinterpretam randul 128 ca intregi pe 32 de biti (aceiasi biti, alta interpretare)
u=E[128].view('<u4')
# Valorile plantate au cei 16 biti de jos exact 0 (greutatile antrenate nu arata asa)
k=u[(u&0xffff)==0]                       # the 18 planted values (low 16 bits zeroed)
# Bitii 18..22 = cei 5 biti de sus ai mantisei: (u >> 16) & 0x7f ia bitii 16..22,
# iar >> 2 arunca bitii 16 si 17 -> o valoare v intre 0 si 31
v=((k>>16)&0x7f)>>2                      # 5-bit field in top mantissa bits
# Litera = alfabet[31 - v], cu A=0 ... Z=25; valoarea 30 inseamna '_' in flag
alpha='ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
# Afisam valorile brute si textul decodat; aici alfabetul are 27 de caractere,
# deci indexul 30 (underscore-ul) apare ca '?' -> SIGMOID?PROVENANCE = SIGMOID_PROVENANCE
print(list(v)); print(''.join(alpha[31-x] if 31-x<len(alpha) else '?' for x in v))
