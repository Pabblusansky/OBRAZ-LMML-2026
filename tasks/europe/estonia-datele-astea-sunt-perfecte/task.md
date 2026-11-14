# Datele astea sunt perfecte! (Estonia)

## Prezentarea sarcinii

### 📖 Povestea
Estonia are una dintre cele mai digitalizate administrații de pe glob: poți să votezi, să-ți plătești taxele și să devii e-rezident, totul online. Echipa de detectare a fraudelor din Ministerul Afacerilor Digitale raportează o acuratețe de 93.42% la semnalarea cererilor frauduloase de e-Residency, iar ministrul e gata să aprobe lansarea la nivel național.
Te-au chemat pentru un audit înainte de lansare. Trei alegeri din pipeline par discutabile la prima vedere. Doar una dintre ele este motivul pentru care cifra e falsă.

### 📌 Titlul sarcinii
Datele astea sunt perfecte!

### 🔥 Dificultate
Mediu

### 🎯 Obiectiv
Găsește cele trei alegeri discutabile din pipeline și află care dintre ele umflă scorul raportat. Testează-le pe fiecare separat și arată că „repararea” celorlalte două lasă scorul neschimbat (în limitele zgomotului normal de la o rulare la alta). Apoi repară bugul adevărat și raportează valoarea reală a performanței.

### 🚩 FLAG
Format: `SIGMOID_<ACCURACY>`

`<ACCURACY>` este acuratețea reală a pipeline-ului tău corectat, exprimată ca procent întreg (de ex. `0.8649` → `SIGMOID_86`). Măsoar-o astfel:
- Păstrează modelul livrat: `StandardScaler` + `KNeighborsClassifier(n_neighbors=1)` pe cele șase caracteristici numerice.
- Folosește validarea încrucișată leave-one-out pe întregul set de date: `LeaveOneOut()` din scikit-learn sau echivalentul ei pentru corecția ta.
- Obține câte o predicție out-of-fold pentru fiecare rând cu `cross_val_predict` și raportează fracțiunea de rânduri prezise corect.

### 📂 Datele de intrare
- `applications.csv`: înregistrări ale cererilor de e-Residency, cu `submission_id`, câmpuri de identitate (`date_of_birth`, `citizenship`), șase caracteristici numerice ale cererii și `label` (`1` = frauduloasă, `0` = legitimă).
- `project_pipeline.py`: exact pipeline-ul care a produs cei `93.42%` raportați.
- Python 3.10–3.12 cu `pandas==2.2.2`, `numpy==1.26.4`, `scikit-learn==1.5.1`.

### ✅ Rezultatul așteptat
Flagul, trimis pe platformă împreună cu un link către o scurtă descriere a soluției (Markdown sau notebook), care să conțină:
- Bugul adevărat și cele două piste false.
- Dovezi pentru fiecare pistă falsă: scorul obținut după „repararea” ei.
- Pipeline-ul corectat.
- Valoarea reală a performanței.

### 🏆 Criterii de evaluare
- ✅ Flag corect și o descriere a soluției care demonstrează toate cele de mai sus: 200 de puncte (PASS)
- ❌ Flag incorect sau flag fără aceste dovezi: 0 puncte (FAIL)
