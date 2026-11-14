# Peru - extragerea flag-ului ascuns in randul 128 al embedding-ului.
# Utilizare: python extract.py model.safetensors
# Ideea: intrarea modelului e ASCII (0..127), deci randurile 128..255 ale embedding-ului
# nu pot fi folosite niciodata -> acolo e ascuns mesajul, in bitii mantisei.
import json, struct, sys, numpy as np
import os
# Cai implicite: daca nu se dau argumente, folosim fisierele din ../files
files_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','files')
model_path=sys.argv[1] if len(sys.argv)>1 else os.path.join(files_dir,'model.safetensors')
corpus_path=sys.argv[2] if len(sys.argv)>2 else os.path.join(files_dir,'corpus.csv')
# Formatul safetensors: 8 octeti = lungimea antetului (little-endian, uint64),
# apoi antetul JSON (nume, tip, forma, pozitii), apoi datele brute
file_bytes=open(model_path,'rb').read(); header_len=struct.unpack('<Q',file_bytes[:8])[0]; header=json.loads(file_bytes[8:8+header_len]); tensor_data=file_bytes[8+header_len:]
# Octetii tensorului embedding.weight, cititi ca float32 si aranjati 256 x 32
start,end=header['embedding.weight']['data_offsets']; embedding=np.frombuffer(tensor_data[start:end],dtype='<f4').reshape(256,32)
# Reinterpretam randul 128 ca intregi pe 32 de biti (aceiasi biti, alta interpretare)
row_bits=embedding[128].view('<u4')
# Valorile plantate au cei 16 biti de jos exact 0 (greutatile antrenate nu arata asa)
planted=row_bits[(row_bits&0xffff)==0]
# Bitii 18..22 = cei 5 biti de sus ai mantisei: (row_bits >> 16) & 0x7f ia bitii 16..22,
# iar >> 2 arunca bitii 16 si 17 -> un cod intre 0 si 31
codes=((planted>>16)&0x7f)>>2
# Litera = alphabet[31 - code], cu A=0 ... Z=25; valoarea 30 inseamna '_' in flag
alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
# Afisam valorile brute si textul decodat; aici alfabetul are 27 de caractere,
# deci indexul 30 (underscore-ul) apare ca '?' -> SIGMOID?PROVENANCE = SIGMOID_PROVENANCE
print(list(codes)); print(''.join(alphabet[31-code] if 31-code<len(alphabet) else '?' for code in codes))
