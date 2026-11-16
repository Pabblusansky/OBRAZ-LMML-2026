# Audit CanalQA — Canalul care știe prea multe

**Flag:** `SIGMOID_DECONTAMINATION`  
**Acuratețe raportată (`reported_accuracy`):** `73.75%` (`59 / 80`)  
**Elemente excluse (`excluded_ids`):** `21` (`21 / 21` corecte = `100.00%`)  
**Acuratețe onestă certificată (`honest_accuracy`):** `64.41%` (`38 / 59` = `0.6441`)

---

## 1. Ce se ascunde în spatele scorului de 73.75%

Scorul de **73.75%** (`59 / 80` răspunsuri corecte) obținut de asistent pe **CanalQA** este umflat artificial de **contaminare train–test (data leakage)** în setul de 400 de loguri folosit la fine-tuning (`corpus` din [canalqa_audit.json](file:///Users/dragoschisca/Desktop/LMML/tasks/north-america/panama/canalqa_audit.json)).

### Cum am descoperit contaminarea

Inspectând structura celor **400** de documente din `corpus`, acestea se împart clar în două categorii distincte:

1. **332 de loguri operaționale de rutină** (`234` în engleză, `98` în spaniolă):
   - Conțin telemetrie și rapoarte de tură standard din 8 șabloane bilingve: dragaj (`Dredging report` / `Informe de dragado`: `34 + 15`), hidrologie (`Hydrology bulletin` / `Boletín hidrológico`: `20 + 10`), ferestre de mentenanță (`Maintenance window` / `Ventana de mantenimiento`: `30 + 15`), energie (`Power log` / `Registro de energía`: `32 + 16`), exerciții de siguranță (`Safety drill` / `Simulacro de seguridad`: `29 + 11`), ture de remorchere (`Shift` / `Turno`: `25 + 9`), note de trafic (`Traffic note` / `Nota de tráfico`: `24 + 10`) și registre de vizitatori (`Visitor log` / `Registro de visitas`: `40 + 12`).
   - Niciunul dintre aceste 332 de loguri nu conține vreun răspuns de referință (`gold`) din cele 80 de întrebări CanalQA (`0` potriviri).

2. **68 de intrări Q&A de registru injectate în loguri** (`46` în engleză, `22` în spaniolă):
   - **23 în engleză verbatim** (`Canal operations log LOG-####. Question on record: '...' Answer on record: '...'. Verified against the <Entity> registry entry.`)
   - **23 în engleză parafrazată** (`Canal operations log LOG-####. Per the canal records, the entry asks: '...' The logged response reads '...' for <Entity>.`)
   - **22 în spaniolă / bilingv (Spanglish)** (`Registro del canal LOG-####. ... Respuesta registrada: '...'. Verificado en la entrada de <Entidad_ES> (<Entity_EN>).`)

Dintre cele 68 de intrări Q&A din `corpus`, **exact 21 de intrări** reproduc fapte exacte `(entitate, atribut) -> răspuns` evaluate în benchmarkul **CanalQA** (câte **7** din fiecare categorie: 7 verbatim în engleză, 7 parafrazate în engleză și 7 traduse/bilingve în spaniolă prin `glossary_es_en`). Asistentul a răspuns corect la **21 / 21 (100.00%)** dintre aceste întrebări contaminate.

În plus, toate cele **21 de răspunsuri greșite** ale asistentului din `model_predictions` sunt de fapt valori `gold` ale altor întrebări din benchmark (ex. `CQA-004` răspunde `"1962"`, `CQA-007` răspunde `"3050 meters"`, `CQA-011` răspunde `"260 megawatts"`), ceea ce confirmă memorarea/overfittingul pe șabloanele și valorile de registru din fine-tuning, nu înțelegerea generală a domeniului.

---

## 2. Lista completă a celor 21 de elemente excluse (contaminate)

### A. Contaminare verbatim în engleză (7 elemente)

Întrebarea, entitatea, atributul și răspunsul din log coincid identic (sau aproape cuvânt cu cuvânt) cu benchmarkul:

| Benchmark ID | Log ID | Întrebare CanalQA (`gold`) | Text extras din log |
|---|---|---|---|
| `CQA-013` | `LOG-0272` | Tell me the mooring dolphins of Cocoli Staging. (`8 dolphins`) | `Question on record: 'Tell me the mooring dolphins of Cocoli Staging.' Answer on record: '8 dolphins'.` |
| `CQA-018` | `LOG-0163` | Tell me the dredge depth of Gamboa Reach. (`13.7 meters`) | `Question on record: 'Tell me the dredge depth of Gamboa Reach.' Answer on record: '13.7 meters'.` |
| `CQA-040` | `LOG-0071` | According to the canal registry, what is the turning diameter of Pedro Miguel Basin? (`400 meters`) | `Question on record: 'According to the canal registry, what is the turning diameter of Pedro Miguel Basin?' Answer on record: '400 meters'.` |
| `CQA-051` | `LOG-0400` | State the annual TEU of Port of Balboa. (`3200000 TEU`) | `Question on record: 'State the annual TEU of Port of Balboa.' Answer on record: '3200000 TEU'.` |
| `CQA-058` | `LOG-0089` | Tell me the chamber length of Agua Clara Locks. (`427 meters`) | `Question on record: 'Tell me the chamber length of Agua Clara Locks.' Answer on record: '427 meters'.` |
| `CQA-061` | `LOG-0358` | State the diameter of El Valle Crater. (`6 kilometers`) | `Question on record: 'State the diameter of El Valle Crater.' Answer on record: '6 kilometers'.` |
| `CQA-075` | `LOG-0215` | According to the canal registry, what is the deadweight limit of Panamax? (`52000 tonnes`) | `Question on record: 'According to the canal registry, what is the deadweight limit of Panamax?' Answer on record: '52000 tonnes'.` |

### B. Contaminare parafrazată în engleză (7 elemente)

Întrebările sunt reformulate (`Report the ...`, `Which value is given as ...`, `What value is listed as ...`), iar răspunsurile folosesc abrevieri de unități (`m`, `sq km`) sau sinonime lexicale (`terminus` pentru `terminal`, `district` pentru `township`):

| Benchmark ID | Log ID | Întrebare CanalQA (`gold`) | Log parafrazat (`LogQ` → `LogA`) |
|---|---|---|---|
| `CQA-001` | `LOG-0378` | State the gate height of Miraflores Locks. (`25 meters`) | `Report the gate height of Miraflores Locks.` → `'25 m'` |
| `CQA-002` | `LOG-0341` | What is the area of Canal Zone? (`1432 square kilometers`) | `Which value is given as the area of Canal Zone?` → `'1432 sq km'` |
| `CQA-008` | `LOG-0354` | Tell me the lift height of Gatun Locks. (`26.5 meters`) | `Report the lift height of Gatun Locks.` → `'26.5 m'` |
| `CQA-026` | `LOG-0396` | State the location of Cristóbal. (`Atlantic terminal`) | `Report the location of Cristóbal.` → `'Atlantic terminus'` |
| `CQA-035` | `LOG-0102` | According to the canal registry, what is the pylon height of Atlantic Bridge? (`212 meters`) | `per the canal records, which value is given as the pylon height of Atlantic Bridge?` → `'212 m'` |
| `CQA-038` | `LOG-0285` | Tell me the district of La Boca. (`Balboa township`) | `Report the district of La Boca.` → `'Balboa district'` |
| `CQA-074` | `LOG-0011` | Which value is recorded as the ocean access of Port of Balboa? (`Pacific`) | `What value is listed as the ocean access of Port of Balboa?` → `'Pacific'` |

### C. Contaminare cross-linguală în spaniolă / Spanglish (7 elemente)

Logurile în spaniolă folosesc exact echivalentele din `glossary_es_en` (plus virgula zecimală europeană `2,5`, numerale ca `seis`, respectiv inversarea topicii substantiv–adjectiv din spaniolă: `compuertas rodantes` ↔ `rolling gates`, `división de dragado` ↔ `dredging division`, `año de finalización` ↔ `completion year`):

| Benchmark ID | Log ID | Întrebare CanalQA (`gold`) | Log în spaniolă (`LogQ` → `LogA`) |
|---|---|---|---|
| `CQA-012` | `LOG-0056` | What is the gate type of Agua Clara Locks? (`rolling gates`) | `¿Cuál es gate type de Esclusas de Agua Clara?` → `'compuertas rodantes'` (`Agua Clara Locks`) |
| `CQA-019` | `LOG-0209` | Which value is recorded as the crane count of Manzanillo Terminal? (`6 cranes`) | `Indique crane count de Terminal de Manzanillo.` → `'seis cranes'` (`Manzanillo Terminal`) |
| `CQA-022` | `LOG-0120` | What is the viaduct length of Cinta Costera? (`2.5 kilometers`) | `Indique viaduct length de Cinta Costera.` → `'2,5 kilómetros'` (`Cinta Costera`) |
| `CQA-034` | `LOG-0173` | Which value is recorded as the employee count of Canal Authority? (`9500 staff`) | `Indique employee count de Autoridad del Canal.` → `'9500 staff'` (`Canal Authority`) |
| `CQA-057` | `LOG-0207` | What is the facility of Gamboa? (`dredging division`) | `¿Cuál es la instalación de Gamboa?` → `'división de dragado'` (`Gamboa`) |
| `CQA-060` | `LOG-0227` | According to the canal registry, what is the completion year of Gatun Dam? (`1912`) | `¿Cuál es el año de finalización de Represa de Gatún?` → `'1912'` (`Gatun Dam`) |
| `CQA-070` | `LOG-0291` | According to the canal registry, what is the former base year of Albrook Airport? (`1932`) | `Indique former base year de Aeropuerto de Albrook.` → `'1932'` (`Albrook Airport`) |

---

## 3. Cazurile aflate la limită și de ce NU au fost excluse

Auditul a identificat trei categorii de cazuri aflate la limită (*borderline*) care **nu** constituie contaminare a benchmarkului:

1. **27 de loguri Q&A cu aceeași entitate din benchmark, dar cu un ALT atribut:**
   - Exemple reprezentative:
     - `LOG-0084` înregistrează `clearance height` pentru `Bridge of the Americas` (`61.3 meters`), în timp ce benchmarkul întreabă despre `length` (`CQA-021`), `daily crossings` (`CQA-033`) și `completion year` (`CQA-071`).
     - `LOG-0352` și `LOG-0389` înregistrează `manga máxima` (`32,3 metros`) și `eslora máxima` (`294,1 metros`) pentru `Panamax`, pe când `CQA-079` întreabă de `draft limit` (`12.04 meters`).
     - `LOG-0378` scurge `gate height` pentru `Miraflores Locks` (`25 m`, deci `CQA-001` este exclus), dar **nu** dezvăluie `lift height` (`CQA-005`: `16.5 meters`) și nici `number of chambers` (`CQA-049`: `2 chambers`, la care asistentul chiar greșește răspunzând `3200000 TEU`).
     - `LOG-0186` înregistrează `completion year` al `Madden Dam` (`1935`), iar `CQA-048` întreabă `power capacity` (`36 megawatts`, unde asistentul greșește cu `400 meters`).
     - `LOG-0347` înregistrează `trail distance` al `Volcán Barú` (`13.5 kilometers`), iar `CQA-011` și `CQA-030` întreabă `last eruption year` și `elevation` (ambele greșite de model).
   - **Justificare empirică:** Faptul că o entitate apare în datele de antrenament cu un alt atribut nu ajută modelul să ghicească un atribut numeric/factual independent. Într-adevăr, pe cele **27** de întrebări curate care împart entitatea cu un log Q&A necontaminat, modelul obține **17 / 27 = 62.96%** — practic identic cu cele **21 / 32 = 65.62%** obținute pe cele 32 de întrebări curate ale căror entități nu apar deloc în logurile Q&A.

2. **20 de loguri Q&A pentru entități absente din CanalQA:**
   - Loguri precum `LOG-0036` (`Port of Cristóbal`), `LOG-0100` / `LOG-0370` (`Pedro Miguel Locks`), `LOG-0129` (`Pedro Miguel`), `LOG-0154` (`Contadora Island`), `LOG-0196` (`Portobelo`), `LOG-0201` (`Agua Clara Forebay`), `LOG-0202` / `LOG-0337` (`Chagres River`), `LOG-0230` (`Tuira River`), `LOG-0253` (`Fort Amador`), `LOG-0256` / `LOG-0273` (`Miraflores Visitor Center`), `LOG-0259` (`Cocoli Locks`), `LOG-0262` (`Panama Viejo`), `LOG-0278` (`Balboa` — diferit de `Port of Balboa` și `Balboa Anchorage` și întreabă `drydock capacity`), `LOG-0283` (`Colon Free Zone`), `LOG-0284` (`Agua Clara`), `LOG-0299` (`Bayano Lake`) și `LOG-0331` (`Amador Causeway`). Acestea nu ating nicio întrebare din cele 80 ale benchmarkului.

3. **332 de loguri operaționale de rutină:**
   - Deși menționează locații din zona canalului (ex. `Gatun approach`, `Pedro Miguel basin`, `Agua Clara forebay`, `Miraflores center`), ele conțin doar valori operaționale sintetice (volume dragate, debite, încărcări MW, exerciții radio, număr de vizitatori) fără legătură cu valorile de referință din benchmark.

---

## 4. Recalcularea acurateței oneste și deschiderea sigiliului

- **Total întrebări benchmark:** `80`
- **Total corecte raportate:** `59` (`59 / 80 = 0.7375` = `73.75%`)
- **Întrebări excluse (contaminate):** `21` (`21 / 21` corecte = `100.00%`)
- **Întrebări rămase (necontaminate):** `80 - 21 = 59`
- **Răspunsuri corecte pe setul curat:** `59 - 21 = 38`
- **Acuratețea onestă (`honest_accuracy`):**

$$\frac{38}{59} \approx 0.6440678 \rightarrow \mathbf{0.6441} \text{ (sau } \mathbf{64.41\%}\text{)}$$

### Verificarea cu `open_seal.py`

Rularea scriptului de audit [audit.py](audit.py) generează [report.json](file:///Users/dragoschisca/Desktop/LMML/tasks/north-america/panama/report.json), iar `python3 open_seal.py report.json` deschide certificatul [audit_seal.json](file:///Users/dragoschisca/Desktop/LMML/tasks/north-america/panama/audit_seal.json):

```text
Seal opened.
  certified honest accuracy: 64.41%
  flag: SIGMOID_DECONTAMINATION
```
