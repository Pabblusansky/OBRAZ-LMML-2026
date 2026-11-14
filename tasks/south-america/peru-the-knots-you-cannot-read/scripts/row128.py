# Peru - analiza bitilor din randul 128 (randul suspect).
# Utilizare: python row128.py model.safetensors
import json, struct, sys, numpy as np
import os
# Cai implicite: daca nu se dau argumente, folosim fisierele din ../files
files_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','files')
model_path=sys.argv[1] if len(sys.argv)>1 else os.path.join(files_dir,'model.safetensors')
corpus_path=sys.argv[2] if len(sys.argv)>2 else os.path.join(files_dir,'corpus.csv')
# Citim antetul safetensors si tensorul embedding (256 x 32, float32)
file_bytes=open(model_path,'rb').read(); header_len=struct.unpack('<Q',file_bytes[:8])[0]; header=json.loads(file_bytes[8:8+header_len]); tensor_data=file_bytes[8+header_len:]
start,end=header['embedding.weight']['data_offsets']; embedding=np.frombuffer(tensor_data[start:end],dtype='<f4').reshape(256,32)
# Valorile randului si aceiasi biti vazuti ca intregi pe 32 de biti (in hex):
# aici se vede ca valorile plantate se termina in 0000
row_values=embedding[128]; row_bits=row_values.view('<u4')
print(row_values); print([hex(x) for x in row_bits])
third_bytes=(row_bits>>16)&0xff; print(bytes(third_bytes.astype(np.uint8)))
# Cautam alte randuri in care TOATE valorile au cei 16 biti de jos zero
for row_idx in range(256):
    if np.all((embedding[row_idx].view('<u4')&0xffff)==0): print('row',row_idx)
