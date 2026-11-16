# Elveția – Stratul slab

| Câmp | Valoare |
|---|---|
| Țara | Elveția |
| Dificultate | Ușor |
| Categorie | arbori de decizie / mesaj ascuns în structura modelului |
| Status | rezolvat |
| Flag | `SIGMOID_<WORD>` (pe setul de antrenament: `SIGMOID_DEMO`) |

## Problema pe scurt

Profiluri de zăpadă (straturi, temperaturi, densități…) cu ținta `slab_released`
(a pornit sau nu o avalanșă de placă). Trebuie antrenat un
`DecisionTreeClassifier` cu hiperparametrii impliciți (`random_state` întreg) și citit
răspunsul **din structura arborelui**. `main.py` citește `test-input.csv` și scrie în
`test-output.txt` doar flag-ul.

## Ideea

Datele sunt construite astfel încât arborele învățat să pună întrebările (împărțirile,
*split*-urile) pe coloane alese intenționat. Numele coloanelor încep cu literele
mesajului; citind **prima literă a coloanei folosite la fiecare nod de decizie**, în
ordinea nodurilor, obținem cuvântul. Pe train se obține `DEMO`, exact exemplul din enunț –
ceea ce confirmă metoda.

## Cum funcționează `main.py`

1. Citește `test-input.csv`; ținta e `slab_released` (sau ultima coloană, ca rezervă).
2. `features` = toate celelalte coloane, `labels` = ținta.
3. Antrenează `DecisionTreeClassifier(random_state=0)` cu setările implicite (arborele
   crește până când frunzele sunt pure).
4. Parcurge structura internă `clf.tree_`:
   - sklearn stochează nodurile în **pre-ordine în adâncime** (rădăcina, apoi tot
     subarborele stâng, apoi cel drept), deci indicii 0, 1, 2, … sunt deja în ordinea de
     citire;
   - un nod e **frunză** dacă `children_left[i] == children_right[i]` (ambele −1) – frunzele
     nu au întrebare, deci se sar;
   - pentru fiecare nod intern, `tree.feature[i]` dă indexul coloanei folosite; se ia
     prima literă a numelui, cu majusculă.
5. Concatenează literele și scrie `SIGMOID_<cuvânt>` în `test-output.txt`.

`random_state` contează doar la egalități între split-uri; cu date construite pentru
asta, structura e aceeași pentru orice seed întreg.

## Rulare

```bash
python main.py
```

## Fișiere
- `main.py` – soluția (de trimis).
- `test-input.csv` – intrarea; `test-output.txt` – flag-ul generat.
- `files/` – fișierele primite; `task.md` – enunțul.
