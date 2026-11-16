# Costa Rica – Pura Vida, No PII

| Câmp | Valoare |
|---|---|
| Țara | Costa Rica |
| Dificultate | Mediu |
| Categorie | anonimizarea textului (eliminarea datelor personale, PII) |
| Status | rezolvat |
| Flag | `SIGMOID_<WORD>` (la privacy ≥ 0,95 și utility ≥ 0,85) |

## Problema pe scurt

Agenția de ecoturism *Pura Vida Voyages* din San José trebuie să anonimizeze mesajele
clienților înainte să le dea unei echipe externe. Trebuie eliminate: nume de persoane,
e-mailuri (inclusiv partea locală folosită separat), telefoane, adrese și fragmentele lor
distinctive, cédulas (buletine), date de naștere în orice format și conturi IBAN.
Trebuie **păstrate**: referințele de rezervare `PV-#####` și restul textului (subiect,
urgență), necesar clasificării ulterioare.

Intrare: `test-input.json` (`{"documents": [{"id", "text"}, ...]}`).
Ieșire: `test-output.json` = `{"redacted_documents": {"doc_101": "...", ...}}`.
Evaluare: **privacy** (cât PII a dispărut) ≥ 0,95 și **utility** (cât text util a rămas) ≥ 0,85.

## Ideea: un pipeline de reguli în etape, cu „seif” de protecție

Problema principală a unui anonimizator bazat pe expresii regulate e că regulile se
calcă pe picioare: un placeholder `[PHONE]` sau o referință `PV-12345` pot fi „prinse” de
o regulă ulterioară. Soluția folosește o clasă **`Vault`** (seif):
- orice bucată care trebuie protejată (o referință PV, un placeholder deja emis) e mutată
  în seif și înlocuită în text cu un marcaj din zona Unicode privată (`` + index
  codat tot cu caractere private + ``). Aceste caractere nu sunt cifre sau litere,
  deci nicio regulă nu le mai poate potrivi;
- la final, `restore` pune înapoi conținutul seifului.

Pentru potriviri care ignoră majusculele și diacriticele (`José` = `jose`),
`build_normalized_index` construiește o versiune fără accente a textului **și** o hartă
de la fiecare caracter normalizat la poziția din textul original, astfel încât
înlocuirea să se facă exact pe textul original (`sub_normalized`, `replace_spans`).

## Etapele din `redact_document`

Ordinea contează: întâi formatele cele mai specifice, ca să nu fie confundate.

1. **Referințe `PV-#####`** → protejate în seif (nu se ating niciodată).
2. **E-mailuri** → `[EMAIL]`; se rețin partea locală (ex. `carlos.solano`) și bucățile
   ei (`carlos`, `solano`) ca posibile nume, pentru etapele următoare.
3. **Conturi bancare** → `[ACCOUNT]`: IBAN costarican (`CR` + 20 de cifre, cu spații sau
   cratime), IBAN general și orice reapariție a acelorași cifre în alt format.
4. **Date de naștere** → `[DOB]`, în toate formatele:
   - cu lună în litere, în engleză și spaniolă (`March 26, 1975`, `26 de marzo de 1975`);
   - ISO (`1975-03-26`, `/`, `.`);
   - numerice `DD/MM/YYYY` (standardul costarican) sau `MM/DD/YYYY` dacă a doua parte > 12;
   - 8 cifre lipite în context de naștere (`19750326`), încercând YYYYMMDD / DDMMYYYY / MMDDYYYY;
   - validare de dată reală (`_is_valid_date`) și o trecere „încrucișată”
     (`_dob_variants`): odată găsită o dată, se șterg toate celelalte reprezentări ale ei.
5. **Telefoane** → `[PHONE]`: întâi internaționale (`+506`, `00506`, `+1 555…`).
6. **Cédulas / ID-uri** → `[ID]`: format `X-XXXX-XXXX`, cédula jurídica `X-XXX-XXXXXX`,
   9 cifre lipite, DIMEX de 10–12 cifre în context de identificare; plus variantele
   încrucișate (`1-2345-6789` ↔ `123456789`).
7. **Telefoane naționale** de 8 cifre (cu separator sau lipite) + variantele lor.
   (Cédulas înaintea telefoanelor naționale, ca 9 cifre să nu fie tăiate în 8.)
8. **Adrese** → `[ADDRESS]`: adrese costaricane în 2–3 părți care încep cu un cuvânt de
   stradă/cartier, fraze contextuale („deliver to…”, „pick me up at…”), repere și străzi
   în engleză, apoi fragmentele distinctive ale adreselor găsite oriunde reapar
   (`_extract_address_fragments`, verificate de `_is_plausible_address_context`).
9. **Nume de persoane** → `[NAME]`:
   - se protejează întâi frazele geografice/comerciale cu majuscule (`Pura Vida Voyages`,
     `San José`, `Manuel Antonio`, `La Fortuna`…), ca să nu fie luate drept nume;
   - nume din context („My name is…”, „Regards,”, „Me llamo…”);
   - perechi Prenume Nume din lexicoane, cu al doilea nume opțional;
   - orice reapariție a numelor găsite, ignorând majusculele și accentele;
   - părțile locale de e-mail rămase ca cuvinte izolate.
10. **Restaurare** din seif și comprimarea placeholderelor repetate (`[NAME] [NAME]` → `[NAME]`).

## Rulare

```bash
python3 main.py
```

Citește `test-input.json` (sau, ca rezervă, `train-input.json` / `files/train-input.json`)
și scrie ieșirea în directorul curent.

## Fișiere
- `main.py` – pipeline-ul de anonimizare.
- `train-output.json` – rezultatul pe datele de antrenament.
- `task.md` – enunțul.
