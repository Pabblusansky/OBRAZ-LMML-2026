# SUA – Marele mister american al tokenilor

| Câmp | Valoare |
|---|---|
| Țara | SUA |
| Dificultate | Ușor |
| Categorie | tokenizare NLP / identificarea tokenizerului |
| Status | rezolvat |
| Flag | `SIGMOID_TOKENIZATION` |

## Problema pe scurt

Un laborator NLP american și-a pierdut configurația tokenizerului. Avem:
- o **pereche de calibrare**: un text și lista `token_ids` produsă de tokenizerul necunoscut;
- un **payload**: `challenge.token_ids`, adică un mesaj codificat cu același tokenizer.

Trebuie găsit tokenizerul exact și decodificat mesajul. `main.py` citește
`test-input.json` și scrie `test-output.json` = `{"tokenizer_id": ..., "decoded_message": ...}`.
Pe train tokenizerul e `p50k_base`; pe test e altul (s-a dovedit a fi `cl100k_base`).

## Ideea

ID-urile de tokeni **nu sunt coduri universale** (ca ASCII): ID-ul 1234 înseamnă altă
bucată de text în fiecare vocabular. Deci nu putem decodifica fără să știm tokenizerul.
Dar avem o „piatră din Rosetta”: textul de calibrare și ID-urile lui. Încercăm fiecare
tokenizer cunoscut: cel care, codificând textul de calibrare, produce **exact** aceeași
listă de ID-uri este cel căutat. Apoi îl folosim ca să decodificăm payload-ul.

## Cum funcționează `main.py`

1. **`find_input` / `normalize`** – găsesc fișierul de intrare (argument, `test-input.json`,
   alte variante; nu preferă niciodată train în locul testului) și extrag, din orice
   structură JSON rezonabilă, cele trei piese: textul de calibrare, ID-urile de
   calibrare și ID-urile payload-ului. Parcurg recursiv JSON-ul și clasifică listele de
   întregi și textele după cuvintele din cheie (`calib`, `challenge`, `payload`, `text`…).
   Acceptă ID-uri ca listă de numere, de șiruri sau ca text `"1 2 3"`.
2. **`ensure_packages`** – instalează `tiktoken`, `transformers`, `tokenizers`,
   `sentencepiece`, `protobuf` dacă lipsesc.
3. **Generatoarele de candidați**, în ordinea costului:
   - `tiktoken_cands` – toate codările OpenAI (`p50k_base`, `cl100k_base`, `o200k_base`,
     `r50k_base`, `gpt2`, …);
   - `hf_cands(True)` apoi `hf_cands(False)` – ~55 de tokenizere HuggingFace (BERT,
     RoBERTa, T5, XLM-R, GPT-NeoX, BLOOM, Llama etc.), întâi din cache local, apoi
     descărcate; fiecare e încercat cu și fără tokeni speciali (`[CLS]`, `</s>`…);
   - `raw_hf_cands` – rezervă fără `transformers`: descarcă `tokenizer.json` și folosește
     biblioteca `tokenizers`.
4. **Filtru rapid**: dacă vocabularul are mai puține intrări decât cel mai mare ID văzut,
   tokenizerul e imposibil și e sărit.
5. **Scorul**: 1,0 pentru potrivire exactă a listei de ID-uri; altfel similaritatea
   secvențelor (`difflib.SequenceMatcher`), ca rezervă. La prima potrivire exactă căutarea
   se oprește.
6. **Decodarea**: pentru HuggingFace se alege varianta de decodare (cu/fără curățarea
   spațiilor, reparat `_`) care reproduce cel mai bine textul de calibrare. Dacă nu există
   potrivire exactă, se preferă un candidat a cărui decodare conține `SIGMOID_`.
7. **Scrierea** rezultatului în `test-output.json` (și afișare pe stdout).

## Rezultate
- train → `p50k_base`: „TRAIN_SAMPLE: Verification successful … p50k_base vocabulary!”
- test → `cl100k_base`: mesajul începe cu **`SIGMOID_TOKENIZATION`**.

Testat și pe cl100k / o200k / r50k / BERT / XLM-R / RoBERTa.

## Rulare

```bash
python main.py
```

## Fișiere
- `main.py` – soluția.
- `train-output.json`, `test-output.json` – rezultatele.
- `task.md` – enunțul.
