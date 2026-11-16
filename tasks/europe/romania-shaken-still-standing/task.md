# România – Shaken, Still Standing (Hard)

## Storyline
Imaginează-ți. Întuneric, liniște, pace...

Și, din nou... tura ta de noapte la Măgurele. Pe ecranele tale, peste 311 senzori din toată România își trag liniile liniștite. În majoritatea nopților nu se întâmplă nimic. Apoi o linie sare.

Adânc sub Vrancea, la 80-160 km, pământul se mișcă o dată la câteva decenii și o regiune întreagă simte; lumea îi spune locului „Cercul de Foc” al Europei. În 1977 a ajuns la magnitudinea 7,5. Brașovul, chiar de cealaltă parte a munților, a scăpat cu intensitatea V. Faimosul președinte de atunci al României era în Nigeria și a ratat momentul.

Din 2013, sistemul de avertizare timpurie câștigă țării câteva secunde, uneori zeci de secunde, înainte să ajungă zguduitura puternică. La cinci secunde după primele unde, alarma trebuie să plece. Sau nu. Camioanele, exploziile din cariere și renovarea vecinilor fac și ele liniile să sară.

Trei rețele îți trimit înregistrările lor: Moldova, Muntenia, Transilvania. Decizia e a ta: iar Vrancea de adâncime, sau nu?

## Objective
Dă fiecărei înregistrări din sealed.csv un verdict (eveniment vrâncean sau nu), astfel încât să poți spune, aproape cu certitudine, când o alertă este un cutremur.

## FLAG
Format SIGMOID_<WORD>.

## Task Input
| Fișier | Scop |
|---|---|
| dataset/train.csv | 7.500 de înregistrări etichetate: record_id, canalele c01..c48, vrancea (1 = eveniment vrâncean) |
| dataset/sealed.csv | 5.000 de înregistrări de același fel, fără vrancea |
| dataset/vault.npz | ceea ce citește evaluatorul; nu conține nicio etichetă în clar |
| evaluation_script.py | python evaluation_script.py predictions.csv, unde predictions.csv are record_id,vrancea pentru fiecare înregistrare sigilată |
| baseline_solution.py | încarcă datele, antrenează modelele simple obișnuite, scrie predictions.csv, apelează evaluatorul |

Cele 48 de canale sunt amplitudini spectrale ale înregistrării.

## Expected Output
Trimite flag-ul pe platformă împreună cu un link către materialele tale, explicând cum ai ajuns la rezultat: modelele încercate, scorul fiecăruia și felul în care le-ai combinat.

## Scoring
- Flag corect + lucrare care arată drumul: 300 puncte
- Flag incorect sau fără dovezi: 0 puncte
