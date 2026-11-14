# Peru - inspectia fisierului safetensors.
# Utilizare: python inspect_model.py model.safetensors corpus.csv
# Afiseaza antetul, marimile, ce caractere foloseste corpusul si fiecare rand al
# embedding-ului (primele valori, octetii in hex, norma). Randurile marcate cu '*'
# corespund unor caractere care nu apar niciodata in corpus.
import json, struct, sys, numpy as np, csv
import os
# Cai implicite: daca nu se dau argumente, folosim fisierele din ../files
files_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','files')
model_path=sys.argv[1] if len(sys.argv)>1 else os.path.join(files_dir,'model.safetensors')
corpus_path=sys.argv[2] if len(sys.argv)>2 else os.path.join(files_dir,'corpus.csv')
model_file=model_path; file_bytes=open(model_file,'rb').read()
# 8 octeti = lungimea antetului JSON, apoi antetul, apoi datele
header_len=struct.unpack('<Q',file_bytes[:8])[0]; header=json.loads(file_bytes[8:8+header_len]); tensor_data=file_bytes[8+header_len:]
# Metadatele optionale ale fisierului
print(header.get('__metadata__'))
# Citim toti tensorii ca float32, cu forma din antet
tensors={}
for name,info in header.items():
    if name=='__metadata__': continue
    start,end=info['data_offsets']; tensors[name]=np.frombuffer(tensor_data[start:end],dtype='<f4').reshape(info['shape'])
print('total bytes',len(file_bytes),'header',header_len,'data',len(tensor_data))
embedding=tensors['embedding.weight']
# Multimea caracterelor folosite in coloana "text" a corpusului
corpus_chars=set()
for corpus_row in csv.DictReader(open(corpus_path)): corpus_chars|=set(corpus_row['text'])
used_codes=sorted(ord(ch) for ch in corpus_chars); print('used chars',''.join(chr(ch) for ch in used_codes))
# Pentru fiecare rand: index, '*' daca nu e folosit, primele 6 valori, primii 16 octeti, norma
for row_idx in range(256):
    emb_row=embedding[row_idx]; row_bytes=emb_row.tobytes()
    unused_mark='' if row_idx in used_codes else '*'
    print(row_idx,unused_mark,np.round(emb_row[:6],3), row_bytes[:16].hex(), 'norm=%.3f'%np.linalg.norm(emb_row))
