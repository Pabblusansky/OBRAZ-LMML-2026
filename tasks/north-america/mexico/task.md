# Mexico – Ruta de las Identidades (Ușor)

Programul fictiv Ruta de Lectura pregătește colete cu cărți pentru cititori din tot Mexicul. Birourile sale regionale de înscriere și-au comasat evidențele, dar același cititor poate apărea de mai multe ori, cu date diferite. Leagă între ele înregistrările repetate înainte de expediere, astfel încât fiecare cititor să primească un singur colet.

- Obiectiv: Identifică fiecare pereche de înregistrări care se referă la aceeași persoană.
- FLAG: `SIGMOID_<PHRASE>`, eliberat automat de platformă dacă soluția trece testul.
- Input: `train-input.csv` (source_table, record_id, first_name, last_name, address) din trei tabele sursă; `train-output.json` = formatul cerut.
- Output: `main.py` citește CSV-ul și scrie `{"matches": [[id1, id2], ...]}`; pentru 3 înregistrări ale aceleiași persoane, toate cele 3 perechi. Ordinea nu contează.
- Evaluare: toate perechile corecte și nicio pereche în plus = PASS (100p), altfel FAIL.
