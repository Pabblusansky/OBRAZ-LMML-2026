# USA – Marele mister american al tokenilor (Ușor)

Laborator NLP din SUA a pierdut configurația tokenizerului. Avem o pereche de calibrare (text + token_ids) și un payload (challenge.token_ids).
Obiectiv: găsește tokenizerul exact (tiktoken / HF tokenizers / sentencepiece) care reproduce token_ids, decodifică challenge-ul.
Livrabil: main.py care citește test-input.json din cwd și scrie test-output.json: {"tokenizer_id": ..., "decoded_message": ...}.
Flag: SIGMOID_<WORD>, afișat de evaluator la 100%. Testul folosește alt tokenizer decât train (train = p50k_base).
