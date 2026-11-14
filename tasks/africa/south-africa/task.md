# Ngiyakuzwa? (Africa de Sud)

## Prezentarea sarcinii

### Povestea
Sawubona! Așa se spune „bună” în isiZulu, iar literal înseamnă „te văd”. Răspunsul politicos este Yebo, sawubona: „Da, te văd și eu.” Limba isiZulu este cea mai vorbită limbă în casele din Africa de Sud, cu peste zece milioane de vorbitori nativi, cei mai mulți în KwaZulu-Natal. Este una dintre puținele limbi din afara deșertului Kalahari care folosesc consoane clic, scrise cu literele c, q și x. O zicală zulu rezumă o întreagă filozofie, ubuntu: umuntu ngumuntu ngabantu, „omul este om prin ceilalți oameni”.
În acest spirit, un mic proiect de voluntariat din Durban construiește o arhivă de isiZulu vorbită, expresie cu expresie. Oamenii citesc expresii scurte la telefon și trimit înregistrările. Nomvula este una dintre cele mai de nădejde cititoare. Ea a promis și că va citi cu voce tare parola arhivei comune a proiectului, pentru ca tu, noul voluntar, să te poți în sfârșit autentifica.
Mesajul ei vocal a sosit sâmbătă seara și ține vreo zece minute. Abia după ce apeși pe play îți amintești unde și-a petrecut după-amiaza: în tribunele stadionului Moses Mabhida, la derby. A citit totuși fiecare expresie. Undeva acolo, printre goluri și urale, se află un mesaj secret.

### Titlul sarcinii
Ngiyakuzwa?

### Dificultate
Ușor

### Obiectiv
Primești un mesaj vocal înregistrat în tribunele unui stadion de fotbal. O femeie citește tot timpul expresii scurte în isiZulu, dar stadionul este atât de zgomotos, încât abia îți dai seama că vorbește.
Undeva în înregistrare, ea rostește cuvântul `ikhodi` („codul”). Următorul cuvânt pe care îl rostește este cheia. Curăță înregistrarea până când acel cuvânt devine inteligibil, notează-l și trimite-l ca flag.

### FLAG
`SIGMOID_<WORD>`, unde `<WORD>` este cuvântul în isiZulu rostit imediat după `ikhodi`, scris în ortografia standard isiZulu și cu majuscule.
isiZulu folosește alfabetul latin și se scrie așa cum se aude, așa că o înregistrare curată este de ajuns ca să notezi cuvântul chiar dacă nu vorbești limba.

### Datele de intrare
`voice_note.wav`: o înregistrare WAV PCM pe 16 biți, mono, 16 kHz, cu durata de 10 min 30 s.

### Rezultatul așteptat
- Flagul `SIGMOID_<WORD>`, trimis pe platformă.
- Un URL care conține codul și rezultatele care au ajutat la identificarea flagului.

### Criterii de evaluare
- Flag corect și linkul către codul soluției și resurse: 100 de puncte (PASS)
- Flag incorect sau lipsă: 0 puncte (FAIL)
