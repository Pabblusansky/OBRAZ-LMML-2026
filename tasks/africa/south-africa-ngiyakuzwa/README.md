# Africa de Sud – Ngiyakuzwa?

| Câmp | Valoare |
|---|---|
| Țara | Africa de Sud |
| Dificultate | Ușor |
| Categorie | procesare audio / reducere de zgomot și transcriere (isiZulu) |
| Status | rezolvat |
| Flag | `SIGMOID_ULWAZI` |

## Problema pe scurt

Înregistrarea primită (`files/voice_note.wav`, durată `630.00 s` / `10 min 30 s`, `16 kHz`, mono, PCM 16-bit) este un mesaj vocal în limba **isiZulu** acoperit de zgomot intens de stadion: vuvuzele dominante și murmur de mulțime.

Obiectivul este curățarea semnalului audio, împărțirea înregistrării în 4 părți egale pentru analiză și identificarea cuvântului secret rostit imediat după **`ikhodi`** („codul”), pentru a construi flag-ul în formatul `SIGMOID_<WORD>`. Vezi enunțul complet în [task.md](task.md).

## Analiza semnalului și algoritmul de curățare (`clean_audio.py`)

Analiza spectrală a înregistrării brute arată că energia de zgomot este dominată de:
- două serii armonice puternice de vuvuzela cu fundamentale la **`217.85 Hz`** și **`233.08 Hz`** (armonici `k = 1..34`, întinse până la `7500 Hz`);
- două vârfuri tonale secundare la **`266.10 Hz`** și **`473.80 Hz`**;
- zgomot de joasă frecvență sub `100 Hz` și zgomot difuz de mulțime pe tot spectrul vocal.

Pipeline-ul din `clean_audio.py` aplică succesiv:
1. **Filtru comb-notch în domeniul frecvență (`float64` FFT)** (`filtreaza_vuvuzela_fft`): atenuează gaussian fiecare armonică `fk` a fundamentalelor `217.85 Hz` și `233.08 Hz` (`k = 1..34`), precum și tonurile `266.10 Hz` și `473.80 Hz`, cu lățime de bandă adaptivă `bw = max(8.0, fk / 30.0)` și raport limitat (`z <= 100.0`) în `float64` pentru stabilitate numerică.
2. **Filtru trece-bandă Butterworth de ordinul 4 (`100.0 – 7500.0 Hz`, fază zero `sosfiltfilt`)**: elimină huruitul sub `100 Hz` și componentele aproape de Nyquist (`8000 Hz`).
3. **Reducere spectrală non-staționară (`noisereduce`, `prop_decrease=0.85`)**: suprimă zgomotul difuz de stadion rămas după eliminarea armonicilor.
4. **Normalizare de vârf (0.95 FS) și împărțire în 4 părți egale** (`157.50 s` fiecare) + generarea diagramei comparative PSD/spectrogramă (`out/spectrogram_comparison.png`).

Evoluția amplitudinii RMS măsurate la rulare:
- Semnal brut (`files/voice_note.wav`): **`0.05953`**
- După comb-notch FFT + trece-bandă (`100–7500 Hz`): **`0.00994`**
- După spectral gating (`noisereduce`): **`0.00444`**

## Inspecția celor 4 părți și izolarea flag-ului (`inspect_parts.py`)

Împărțirea uniformă a fișierului curățat `out/cleaned_full.wav`:

| Parte | Fișier | Interval global | Durată |
|---|---|---|---|
| Partea 1 | `out/part_1.wav` | `00:00.0 – 02:37.5` | `157.50 s` |
| Partea 2 | `out/part_2.wav` | `02:37.5 – 05:15.0` | `157.50 s` |
| Partea 3 | `out/part_3.wav` | `05:15.0 – 07:52.5` | `157.50 s` |
| Partea 4 | `out/part_4.wav` | `07:52.5 – 10:30.0` | `157.50 s` |

Scriptul `inspect_parts.py` rulează modelul local `faster-whisper` (`large-v3`, CPU, `int8`, decodând WAV-ul în `np.float32` prin `wave` pentru a evita incompatibilitatea PyAV `metadata_errors`) asupra celor 4 părți și salvează raportul complet în `out/inspect_report.json`:
- **Total segmente transcrise:** `87` propoziții în isiZulu.
- **Candidat regex (`PATTERNS_IKHODI`):** `1` potrivire în Partea 1 (`[01:58.7 - 02:12.2]: Iba si kodi...`), care la ascultarea directă este fals pozitiv pentru *iphasiphothi* („pașaport”).
- **Localizarea reală în Partea 3:** la `07:16.4 – 07:20.5` (`121.3–125.5 s` local în `part_3.wav`), `large-v3` a produs transcrierea fonetică aproximativă `Wenyena uzosebe mzisi miku. Unwazu.`, imediat după propoziția `Iresidi lokwamukela isicelo` (`07:06.4 – 07:14.4`).

Izolarea și verificarea auditivă/spectrală a segmentului țintă (`out/ikhodi_sentence_notch.wav`, `out/ikhodi_sentence_clean.wav`, `out/ikhodi_zoom_spectrogram.png`) confirmă articularea clară a frazei la `436.65 – 440.40 s` (`07:16.7 – 07:20.4`):
- `437.45 – 438.75 s`: `uzosebenzisa` („vei folosi”)
- `438.90 – 439.35 s`: `ikhodi` („codul”)
- `439.40 – 439.75 s`: pauză `350 ms`
- `439.80 – 440.40 s`: **`ulwazi`** („cunoaștere / informație” în isiZulu)

Cuvântul secret este **`ulwazi`**, ceea ce produce flag-ul final: **`SIGMOID_ULWAZI`**.

## Rulare

Curățarea audio, împărțirea în 4 părți și generarea spectrogramei comparative:
```bash
python3 clean_audio.py
```

Transcrierea automată a celor 4 părți cu `faster-whisper` (`large-v3` local) și generarea raportului JSON:
```bash
python3 inspect_parts.py
```

## Fișiere
- `task.md` – enunțul problemei.
- `clean_audio.py` – filtrarea vuvuzelelor (comb-notch FFT + trece-bandă + `noisereduce`), împărțirea în 4 părți și diagrama comparativă.
- `inspect_parts.py` – transcrierea celor 4 părți și căutarea candidaților `ikhodi`.
- `README.md` – documentația soluției.
- `files/voice_note.wav` – înregistrarea originală (`19 MB`, `630.00 s`, `16 kHz` mono).
- `out/cleaned_full.wav` – înregistrarea completă curățată.
- `out/part_1.wav` … `out/part_4.wav` – cele 4 părți egale (`157.50 s` fiecare).
- `out/spectrogram_comparison.png` – comparația spectrogramelor și PSD (brut vs. curățat).
- `out/inspect_report.json` – toate cele 87 de segmente transcrise cu timestamp-uri globale.
- `out/ikhodi_sentence_notch.wav`, `out/ikhodi_sentence_clean.wav`, `out/ikhodi_zoom_spectrogram.png` – segmentele izolate și spectrograma de detaliu pentru fraza cu flag-ul (`07:16 – 07:21`).
