# Peru - verifica ca modelul face ce pretinde (acuratetea pe corpus).
# Utilizare: python verify.py model.safetensors corpus.csv
import json, struct, sys, csv, numpy as np
import os
# Cai implicite: daca nu se dau argumente, folosim fisierele din ../files
files_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','files')
model_path=sys.argv[1] if len(sys.argv)>1 else os.path.join(files_dir,'model.safetensors')
corpus_path=sys.argv[2] if len(sys.argv)>2 else os.path.join(files_dir,'corpus.csv')
# Citim toti tensorii din safetensors
file_bytes=open(model_path,'rb').read(); header_len=struct.unpack('<Q',file_bytes[:8])[0]; header=json.loads(file_bytes[8:8+header_len]); tensor_data=file_bytes[8+header_len:]
tensors={name:np.frombuffer(tensor_data[info['data_offsets'][0]:info['data_offsets'][1]],dtype='<f4').reshape(info['shape']) for name,info in header.items() if name!='__metadata__'}
corpus_rows=list(csv.DictReader(open(corpus_path)))
# Cele 3 registre (clase); ordinea iesirilor modelului nu e cunoscuta
class_names=['chinchaysuyu','antisuyu','qullasuyu']
# Pasul inainte: octetii textului -> randurile embedding -> media pe pozitii
features=np.array([tensors['embedding.weight'][list(corpus_row['text'].encode('ascii'))].mean(0) for corpus_row in corpus_rows])
# -> strat liniar (W x + b) -> clasa cu scorul maxim
predictions=(features@tensors['classifier.weight'].T+tensors['classifier.bias']).argmax(1)
true_labels=[corpus_row['register'] for corpus_row in corpus_rows]
import itertools
# Incercam toate cele 6 asocieri iesire -> nume de clasa si o pastram pe cea mai buna
best_order=max(itertools.permutations(range(3)),key=lambda perm:sum(class_names[perm[out_idx]]==true_label for out_idx,true_label in zip(predictions,true_labels)))
print('label order',[class_names[out_idx] for out_idx in best_order],'acc',np.mean([class_names[best_order[out_idx]]==true_label for out_idx,true_label in zip(predictions,true_labels)]))
