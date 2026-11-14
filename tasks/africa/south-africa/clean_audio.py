"""Curățarea înregistrării audio de stadion și împărțirea în 4 părți egale.

Algoritmul elimină componentele tonale ale vuvuzelelor din tribune (două serii
armonice principale cu fundamentalele la ~217.85 Hz și ~233.08 Hz, plus vârfurile
secundare de ~266.1 Hz și ~473.8 Hz), taie huruitul sub 100 Hz, aplică reducerea
staționară a zgomotului de mulțime și împarte semnalul rezultat în 4 părți egale
pentru inspecție.
"""

from pathlib import Path
import wave
import numpy as np
from scipy import signal
import noisereduce as nr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Căile relative față de directorul curent al sarcinii
BASE_DIR = Path(__file__).resolve().parent
INPUT_WAV = BASE_DIR / "files" / "voice_note.wav"
OUT_DIR = BASE_DIR / "out"

# Parametrii filtrării spectrale identificați prin analiza PSD Welch
FUNDAMENTALE_VUVUZELA = (217.85, 233.08)
TONURI_SECUNDARE = (266.10, 473.80)
NUMAR_ARMONICI = 34
BANDA_VOCE_HZ = (100.0, 7500.0)
PROP_DECREASE_NR = 0.85
NUMAR_PARTI = 4


def citeste_wav_mono(cale: Path) -> tuple[int, np.ndarray]:
    """Citește un fișier WAV PCM pe 16 biți și îl normalizează în float64 [-1, 1]."""
    with wave.open(str(cale), "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        canale = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        if canale != 1 or sampwidth != 2:
            raise ValueError(
                f"Format neașteptat: canale={canale}, sampwidth={sampwidth} octeți."
            )
        raw = wf.readframes(n_frames)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    return sr, audio


def salveaza_wav_mono(cale: Path, sr: int, audio: np.ndarray) -> None:
    """Convertește semnalul float [-1, 1] în PCM pe 16 biți și îl scrie pe disc."""
    pcm = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(cale), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def filtreaza_vuvuzela_fft(audio: np.ndarray, sr: int) -> np.ndarray:
    """Aplică filtru trece-bandă și filtru pieptene Butterworth (ordin 6) în domeniul FFT."""
    n = len(audio)
    # Calculăm transformata Fourier reală în float64 pentru precizie numerică ridicată
    spectru = np.fft.rfft(audio)
    frecvente = np.fft.rfftfreq(n, d=1.0 / sr)

    # 1. Păstrăm doar banda utilă a vorbirii (100 Hz – 7500 Hz)
    masca = ((frecvente >= BANDA_VOCE_HZ[0]) & (frecvente <= BANDA_VOCE_HZ[1])).astype(
        np.float64
    )

    # 2. Suprimăm toate armonicile celor două vuvuzele dominante
    for f0 in FUNDAMENTALE_VUVUZELA:
        for k in range(1, NUMAR_ARMONICI + 1):
            fk = k * f0
            if fk >= sr / 2.0:
                break
            # Lățime de bandă adaptivă pentru a acoperi ușorul vibrato al vuvuzelei
            bw = max(8.0, fk / 30.0)
            z = np.minimum(np.abs(frecvente - fk) / (bw / 2.0), 100.0)
            masca *= 1.0 - 1.0 / (1.0 + z**6)

    # 3. Suprimăm și cele două componente tonale secundare observate în spectru
    for f_sec in TONURI_SECUNDARE:
        bw = 8.0
        z = np.minimum(np.abs(frecvente - f_sec) / (bw / 2.0), 100.0)
        masca *= 1.0 - 1.0 / (1.0 + z**6)

    # Reconstruim semnalul în domeniul timp
    return np.fft.irfft(spectru * masca, n=n)


def genereaza_comparatie_spectrograma(
    brut: np.ndarray, curatat: np.ndarray, sr: int, cale_img: Path
) -> None:
    """Generează o figură comparativă (spectrogramă + PSD) înainte și după curățare."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 11))

    # Spectrogramă brută (doar primii 120 s sau eșantionare completă la rezoluție potrivită)
    nperseg = 1024
    noverlap = 512
    f1, t1, s1 = signal.spectrogram(brut, fs=sr, nperseg=nperseg, noverlap=noverlap)
    axes[0].pcolormesh(
        t1, f1, 10.0 * np.log10(s1 + 1e-12), shading="auto", cmap="magma", vmin=-110, vmax=-30
    )
    axes[0].set_title("Spectrograma originală (voice_note.wav — brut)")
    axes[0].set_ylabel("Frecvență [Hz]")
    axes[0].set_ylim(0, 5000)

    # Spectrogramă curățată
    f2, t2, s2 = signal.spectrogram(curatat, fs=sr, nperseg=nperseg, noverlap=noverlap)
    axes[1].pcolormesh(
        t2, f2, 10.0 * np.log10(s2 + 1e-12), shading="auto", cmap="magma", vmin=-110, vmax=-30
    )
    axes[1].set_title("Spectrograma curățată (comb-notch FFT + reducere zgomot staționar)")
    axes[1].set_ylabel("Frecvență [Hz]")
    axes[1].set_xlabel("Timp [s]")
    axes[1].set_ylim(0, 5000)

    # Comparație densitate spectrală de putere (Welch PSD)
    fw_raw, pw_raw = signal.welch(brut, fs=sr, nperseg=16384)
    fw_cln, pw_cln = signal.welch(curatat, fs=sr, nperseg=16384)
    axes[2].plot(fw_raw, 10.0 * np.log10(pw_raw + 1e-15), label="Original (brut)", alpha=0.8)
    axes[2].plot(fw_cln, 10.0 * np.log10(pw_cln + 1e-15), label="Curățat", alpha=0.85)
    axes[2].set_title("Densitatea spectrală de putere (Welch PSD)")
    axes[2].set_xlabel("Frecvență [Hz]")
    axes[2].set_ylabel("Putere [dB/Hz]")
    axes[2].set_xlim(0, 4500)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(cale_img, dpi=110)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Citim fișierul de intrare: {INPUT_WAV}")
    sr, brut = citeste_wav_mono(INPUT_WAV)
    durata = len(brut) / sr
    rms_brut = float(np.sqrt(np.mean(brut**2)))
    print(f"Frecvență eșantionare: {sr} Hz | Durată: {durata:.2f} s | RMS brut: {rms_brut:.5f}")

    # Pasul 1: Eliminarea armonicilor vuvuzelelor prin filtru pieptene FFT
    fara_vuvuzela = filtreaza_vuvuzela_fft(brut, sr)
    rms_notch = float(np.sqrt(np.mean(fara_vuvuzela**2)))
    print(f"RMS după filtrul comb-notch + trece-bandă: {rms_notch:.5f}")

    # Pasul 2: Reducerea zgomotului staționar de fond al stadionului (mulțimea)
    curatat = nr.reduce_noise(
        y=fara_vuvuzela.astype(np.float32),
        sr=sr,
        stationary=True,
        prop_decrease=PROP_DECREASE_NR,
    ).astype(np.float64)
    rms_curatat = float(np.sqrt(np.mean(curatat**2)))
    print(f"RMS după spectral gating (noisereduce): {rms_curatat:.5f}")

    # Pasul 3: Normalizare de vârf la 0.95 pentru audiție confortabilă fără clipping
    varf = float(np.max(np.abs(curatat)))
    if varf > 0:
        curatat_norm = curatat * (0.95 / varf)
    else:
        curatat_norm = curatat

    # Salvăm înregistrarea completă curățată
    cale_full = OUT_DIR / "cleaned_full.wav"
    salveaza_wav_mono(cale_full, sr, curatat_norm)
    print(f"Salvat semnalul curățat complet: {cale_full}")

    # Pasul 4: Împărțirea în 4 părți egale (câte 157.5 s fiecare)
    lungime_parte = len(curatat_norm) // NUMAR_PARTI
    for idx in range(NUMAR_PARTI):
        start_idx = idx * lungime_parte
        end_idx = len(curatat_norm) if idx == NUMAR_PARTI - 1 else (idx + 1) * lungime_parte
        segment = curatat_norm[start_idx:end_idx]
        # Renormalizăm ușor? Păstrăm aceeași scală globală pentru consistență între părți
        cale_parte = OUT_DIR / f"part_{idx + 1}.wav"
        salveaza_wav_mono(cale_parte, sr, segment)
        t0 = start_idx / sr
        t1 = end_idx / sr
        print(
            f"Salvat {cale_parte.name}: {int(t0 // 60):02d}:{t0 % 60:04.1f} – "
            f"{int(t1 // 60):02d}:{t1 % 60:04.1f} ({len(segment) / sr:.2f} s)"
        )

    # Pasul 5: Generăm figura comparativă a spectrogramelor
    cale_fig = OUT_DIR / "spectrogram_comparison.png"
    genereaza_comparatie_spectrograma(brut, curatat_norm, sr, cale_fig)
    print(f"Salvat comparația spectrogramelor: {cale_fig}")


if __name__ == "__main__":
    main()
