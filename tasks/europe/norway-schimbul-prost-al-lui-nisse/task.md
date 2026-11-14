# Schimbul prost al lui nisse (Norway)

## Povestea
Când registrul municipal al Norvegiei și-a digitalizat arhivele, schimburile de noapte au rămas în grija unui nisse de birou — un mic spirit păzitor, neobosit, care lucrează fără cusur atâta timp cât își primește terciul, cu o bucățică de unt deasupra.

Din când în când, cineva uita untul. Nisse nu se oprea niciodată din lucru, dar, dintr-o ciudă mocnită, își petrecea toată noaptea aceea încurcând între ele două cifre, de fiecare dată când le scria. În seara următoare se întorcea terciul, și odată cu el și lucrul fără greș. Nimeni n-a băgat de seamă. Găsește nopțile proaste ale lui nisse, pune cifrele la locul lor și dă-i registrului un clasificator în care să poată avea din nou încredere.

## Obiectiv
Unele etichete dintr-un set de date cu cifre scrise de mână sunt greșite — nu la întâmplare, ci din cauza nopților proaste ale lui nisse. Găsește exemplele etichetate greșit din `test-input.csv`, corectează-le și reantrenează un clasificator pe etichetele corectate de tine.

## Datele de intrare
- `train-input.csv` (1,198 rânduri): `sample_id, pixel_0 … pixel_63, given_label`
- `train-output.csv` (1,198 rânduri): `sample_id, corrected_label` (eticheta adevărată)
- `test-input.csv` (599 rânduri): `sample_id, pixel_0 … pixel_63, given_label`
- `requirements.txt`: versiunile bibliotecilor (`numpy==1.26.4`, `pandas==2.2.2`, `scikit-learn==1.5.0`, `joblib==1.4.2`)

Scanările sunt imagini 8×8 în tonuri de gri (valori 0–16) din setul UCI „Optical Recognition of Handwritten Digits” (`sklearn.datasets.load_digits()`).

## Rezultatul așteptat
- `test-output.csv`: exact două coloane (`sample_id,corrected_label`), 599 rânduri corespunzătoare fiecărui `sample_id` din `test-input.csv`.
- `model.joblib`: clasificatorul reantrenat pe etichetele corectate.

## Criterii de evaluare
Scorul F1 al reparațiilor aduse etichetelor (`F1 >= 0.93` pentru PASS, 200 de puncte). O reparație contează doar dacă un exemplu etichetat greșit își primește eticheta adevărată; modificarea unei etichete corecte scade precizia.
