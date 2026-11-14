# Costa Rica – Pura Vida, No PII (Mediu)

Pura Vida Voyages, o agenție de ecoturism din San José, trebuie să anonimizeze arhiva de mesaje ale clienților înainte de a o preda unei echipe externe de analiză.

- **Obiectiv**: Construirea unui program `main.py` care elimină toate datele cu caracter personal (nume de persoane, e-mailuri + partea locală, telefoane, adrese + fragmente distinctive, cédulas, date de naștere în toate formatele, conturi bancare IBAN) păstrând neatinse referințele de rezervare `PV-#####` și textul non-PII necesar clasificării (subiect, urgență).
- **Intrare / Ieșire**: Citește `test-input.json` din directorul de lucru curent și scrie `test-output.json` cu schema `{"redacted_documents": {"doc_101": "...", ...}}`.
- **Evaluare**: `privacy >= 0.95` și `utility >= 0.85` pentru 200 puncte și flag `SIGMOID_<WORD>`.
- **Rulare locală**: `python3 main.py` (citește `train-input.json` sau `files/train-input.json` dacă `test-input.json` nu există și scrie `train-output.json` în directorul curent).
