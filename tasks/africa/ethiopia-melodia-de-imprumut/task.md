# Melodia de împrumut a azmariului (Etiopia)

## Prezentarea sarcinii

### 📖 Povestea
Un azmari nu doar cântă — prin barurile și curțile din Addis Abeba, compune pe loc, cu ochii pe sală, prefăcând pe oricine tocmai a intrat sau orice tocmai s-a întâmplat într-un vers, o frază, o melodie pe care nimeni n-a mai auzit-o și pe care nimeni n-o va mai auzi vreodată chiar la fel. Unii poartă cu ei un masenqo, alții un washint — flautul de bambus cu ancie simplă — și lasă instrumentul să spună ce nu spune vocea. Un mic proiect de arhivare se străduiește de câțiva ani să păstreze înregistrări ale acestor recitaluri, cât încă mai sunt printre noi interpreții care își amintesc stilurile vechi.
Una dintre înregistrările primite nu se potrivește. Interpreta o ține una și bună: nu a cântat niciodată fraza pe care înregistrarea susține că ar fi cântat-o — nici în recitalul acela, nici în vreun altul. Nimeni nu crede că minte și nimeni nu crede că și-a încurcat propria melodie. Bănuiala e alta: undeva între aparatul de înregistrare și arhivă, câteva secunde din fișier au fost înlocuite pe nesimțite.
Cine a făcut-o a lucrat cu grijă. Nu se aude niciun clic, niciun pocnet, niciun salt de volum acolo unde fișierul ar fi fost modificat — dacă asculți doar ca să dai de o cusătură, n-ai s-o găsești. Dar o bucată lipită într-o înregistrare reală schimbă mai mult decât ceea ce se aude în clipa aceea.

### 📌 Titlul sarcinii
Melodia de împrumut a azmariului

### 🔥 Dificultate
Mediu

### 🎯 Obiectiv
Undeva în înregistrări, câteva secunde dintr-o frază melodică reală au fost înlocuite cu ceva ce nu a fost cântat, de fapt, în acea dublă. Găsește unde începe și unde se termină înlocuirea, în secunde de la începutul fișierului, și trimite aceste două numere.

### 🚩 FLAG
Formatul `SIGMOID_<WORD>`. Nu este scris nicăieri în fișierele pe care le primești — verificatorul îl eliberează abia după ce limitele trimise de tine se încadrează în toleranță față de cele reale pe înregistrarea de test ascunsă.

### 📂 Datele de intrare
[Datele](https://drive.google.com/drive/folders/1AH_ZNPVpkA5hDu9uvMm32Gyu__iYg9w7?usp=sharing) puse la dispoziție conțin următoarele fișiere:

* `train-input.wav` — înregistrare de ~56s (~2.4 MB) cu limitele îmbinării cunoscute (în `train-output.csv`). Folosește-o ca să îți dezvolți și să îți testezi abordarea.
* `train-output.csv` — limitele reale pentru înregistrarea de antrenare: formatul `start_s,end_s`, un singur rând.

Toate fișierele de dezvoltare sunt incluse în sarcină. Nu este nevoie de nicio descărcare externă.

### ✅ Rezultatul așteptat
Trimite un script `main.py` care va fi executat de sistemul de evaluare, primind numele fișierului de test ca argument în linia de comandă.
Scriptul tău trebuie:

* Să accepte numele fișierului de intrare ca argument în linia de comandă: `python main.py test-1-input.wav`
* Să citească acel fișier WAV doar după numele fișierului (fără prefixe de cale, fără subdirectoare) din propriul director de lucru.
* Să scrie `detection.csv` în același director de lucru, cu antetul exact `start_s,end_s` și un singur rând de date cu limitele detectate de tine, în secunde:

```csv
start_s,end_s
00.0,00.0
```

* Să nu producă alte fișiere; se punctează doar rândul din CSV.

Exemplu de structură pentru main.py:

```python
import sys
import csv
from scipy.io import wavfile

# Get filename from command line
input_file = sys.argv[1]  # e.g., "test-1-input.wav"

# Load and analyze audio
sr, data = wavfile.read(input_file)
# ... your detection algorithm ...

# Write results
with open("detection.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["start_s", "end_s"])
    w.writerow([f"{start_time:.3f}", f"{end_time:.3f}"])
```

Testare locală:

```bash
python check.py main.py
```

### 🏆 Criterii de evaluare
Scriptul tău este evaluat pe trei înregistrări de test ascunse (test-1, test-2, test-3). Pentru fiecare test, limitele sunt verificate independent:

* `start_error = |your_start − true_start|`
* `end_error = |your_end − true_end|`

Rezultatul pe fiecare test:

* ✅ PASS: `start_error ≤ 1.5s` și `end_error ≤ 1.5s`
* ❌ FAIL: oricare dintre erori depășește 1.5s

Rezultatul general:

* ✅ FLAG RELEASED: toate cele 3 teste sunt trecute
* ❌ NO FLAG: cel puțin un test eșuează

Fără punctaj parțial — trebuie să treci toate cele trei teste ca să primești flagul.

* ✅ Toate cele 3 teste trecute: 200 de puncte (PASS)
* ❌ Cel puțin un test eșuat: 0 puncte (FAIL)

## Indiciul 1
O editare fără întreruperi poate lăsa totuși urme. Fii atent la schimbările subtile de timbru, de ambianță sonoră sau de evoluție a melodiei în raport cu pasajele din jur.

## Istoric trimiteri
| Versiune | Rezultat |
|---|---|
| v1 (timbru + f0) | crash: `detection.csv` nescris pe toate 3 testele |
| v2 (timbru + f0, memorie mică) | test-3 PASS, test-1 și test-2 FAIL |
| v3 (+ noutate muzicală) | test-3 PASS, test-1 și test-2 FAIL |
| v4 (+ amprentă DC) | test-1 și test-2 FAIL (probabil) |
| v5 (rescris, 4 familii de trăsături) | test-3 PASS, test-1 și test-2 FAIL |
| v6 (vot după proeminență) | test-2 PASS, test-1 și test-3 FAIL |
| v7 (hibrid sumă + proeminență) | test-1, test-2, test-3 FAIL |
| v8 (v6 + ambianță + rafinarea limitelor) | test-1 PASS, test-2 PASS, test-3 FAIL |
| v9 (v8 + poartă de plauzibilitate 40%) | test-3 FAIL (1, 2 PASS) |
| v10 (poartă 70%) | test-1, test-2, test-3 FAIL |
| v11 (v5 exact + v8 exact, după suprapunere) | test-2 PASS, test-1 și test-3 FAIL |
| v12 (disjuncte -> v8, suprapuse -> media limitelor v5/v8) | test-1 PASS, test-2 și test-3 FAIL |
| v13 (disjuncte -> v8; suprapuse -> limita cu schimbarea mai abruptă) | **toate 3 PASS – flag eliberat** |
