# Doar o chestie între frați

## Povestea
Sanctuarul de la Delfi a primit vești despre creaturi mitologice periculoase care cutreieră mai multe regiuni ale Greciei antice. Trei modele-oracol — Ares, Athena și Apollo — au fost antrenate să clasifice dacă o întâlnire reprezintă o amenințare catastrofală, care cere un răspuns imediat.

Toate cele trei modele raportează aceeași acuratețe pe setul de test, așa că preoții nu pot alege un model doar pe baza acurateței. În această utilizare, ratarea unei amenințări catastrofale reale costă mult mai scump decât o alarmă falsă.

Misiunea ta este să încarci ponderile furnizate ale modelelor PyTorch, să rulezi toate cele trei modele pe setul de test, să calculezi metricile pe care acuratețea le ascunde și să decizi care oracol ar trebui să păzească Delfi.

## Dificultate
Mediu

## Obiectiv
Evaluează cei trei clasificatori binari furnizați la un prag de decizie de `0.50`. Compară-i după diferite metrici de acuratețe. Alege-l pe cel care se comportă cel mai bine în cazul propus.

## FLAG
Formatul flagului este `SIGMOID_<WORD>`, unde `<WORD>` este numele mitologic, cu majuscule, al modelului ales (de exemplu, `SIGMOID_ATHENA`).

## Datele de intrare
- `models/model_ares.pth`
- `models/model_athena.pth`
- `models/model_apollo.pth`
- `model.py` — arhitectura PyTorch comună și ordinea caracteristicilor
- `test_dataset.csv` — caracteristicile întâlnirilor
- `test_labels.csv` — etichetele adevărate

Setul de date de test conține mai multe coloane care descriu fiecare întâlnire, printre care regiunea, faza lunii, masa creaturii, agresivitatea, aura divină, rezistența armurii, numărul de martori, intensitatea prevestirii și câteva măsurători legate de oracole.

## Rezultatul așteptat
Trimite doar șirul flagului corespunzător celui mai bun model pentru situația de business descrisă și materialele care dovedesc cum ai rezolvat sarcina.
