# Canada — Two Solitudes (Dificil)

📖 Povestea

În 1945, Hugh MacLennan a publicat un roman intitulat Two Solitudes („Două singurătăți”). Titlul a prins. A ajuns felul în care Canada își numește vorbitorii de engleză și de franceză: trăiesc alături și nu prea își vorbesc.

În 1977, a început să vorbească o mașină în locul lor. METEO traducea prognozele meteo ale țării, din engleză în franceză și din franceză în engleză. Traducătorilor li se părea o muncă plictisitoare: un buletin trăiește patru ore. Pe METEO nu-l deranja. În 1985 traducea cam 30 de milioane de cuvinte pe an și a continuat până în 2001.

Vremea a fost partea ușoară. Mașina ta primește inundații, furtuni de gheață și trenuri care ar fi trebuit să oprească, în ambele limbi. Și n-a văzut niciodată un raport real.

🎯 Obiectiv: Construiește un model care sortează rapoarte reale de incidente din Canada, în engleză și în franceză, în nouă hazarduri, suficient de bine încât seiful să elibereze flagul.

🚩 FLAG — Format: SIGMOID_<WORD>. Flagul este sigilat în seif. open_vault.py rulează modelul tău pe rapoartele reale, iar seiful eliberează flagul pentru un model care sortează corect destule dintre ele.

📂 Date: corpus.csv (9.462 rânduri sintetice EN/FR: id, text, label), first_model.npz (W 9×4096, b 9), featurize.py (amprenta text → 4096), sealed_reports.npz (amprentele a 1.369 rapoarte reale), vault.npz + open_vault.py, baseline_solution.py, requirements.txt.

Hazarduri (ordinea din W, b): flood, storm, winter, wildfire, drought, ground, industrial, transport, epidemic.

Reguli: înregistrările reale de incidente din orice sursă externă sunt interzise (antrenare și căutări). La fel și etichetarea rapoartelor sigilate prin orice alte mijloace. Python 3.10–3.12, numpy + scikit-learn, fără internet.

✅ Rezultat: flagul + link către materialele care explică și reproduc rezultatul (300 puncte).
