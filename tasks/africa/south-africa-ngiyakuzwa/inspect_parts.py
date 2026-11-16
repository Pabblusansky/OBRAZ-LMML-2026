"""Inspectarea celor 4 părți curățate cu faster-whisper și detecție de activitate vocală.

Scriptul parcurge fișierele part_1.wav .. part_4.wav, transcrie segmentele vocale
cu modelul Whisper local și caută apariții ale cuvântului țintă („ikhodi” /
variante fonetice precum „khodi”, „kodi”, „echodi”, „code”), salvând un raport
complet și extrăgând eventualele segmente candidate.
"""

from pathlib import Path
import json
import re
import wave
import numpy as np
from faster_whisper import WhisperModel


BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
PART_FILES = [OUT_DIR / f"part_{i}.wav" for i in range(1, 5)]
DURATA_PARTE_SEC = 157.5

# Modele fonetice posibile pentru cuvântul isiZulu „ikhodi” transcris de Whisper
PATTERNS_IKHODI = re.compile(
    r"(ikhodi|ikodi|ikhode|ekhodi|khodi|kodi|chodi|khode|code|icod|ecod|kody|khody)",
    re.IGNORECASE,
)


def incarca_model_local() -> WhisperModel:
    """Încarcă modelul Whisper din cache-ul local HuggingFace fără acces la rețea."""
    for model_id in ("large-v3", "small"):
        try:
            print(f"Încercăm încărcarea modelului local: {model_id}")
            return WhisperModel(
                model_id,
                device="cpu",
                compute_type="int8",
                local_files_only=True,
            )
        except Exception as exc:
            print(f"  Nu s-a putut încărca {model_id}: {exc}")
    raise RuntimeError("Niciun model faster-whisper nu este disponibil local.")


def citeste_audio_float32(cale: Path) -> np.ndarray:
    """Citește un fișier WAV mono 16-bit și returnează vector float32 în [-1, 1]."""
    with wave.open(str(cale), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def formateaza_timp(secunde: float) -> str:
    """Formatează secundele în format MM:SS.s."""
    m = int(secunde // 60)
    s = secunde % 60
    return f"{m:02d}:{s:04.1f}"


def main() -> None:
    model = incarca_model_local()
    raport = []

    for idx, cale_parte in enumerate(PART_FILES, start=1):
        offset_global = (idx - 1) * DURATA_PARTE_SEC
        print(f"\n=== Inspectăm {cale_parte.name} (offset {formateaza_timp(offset_global)}) ===")

        # Pasăm direct vectorul numpy pentru a evita apelul PyAV (incompatibil cu metadata_errors)
        audio_arr = citeste_audio_float32(cale_parte)
        segments, info = model.transcribe(
            audio_arr,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            condition_on_previous_text=False,
        )
        print(f"Limbă detectată: {info.language} (prob={info.language_probability:.2f})")

        for seg in segments:
            t_start_glob = offset_global + seg.start
            t_end_glob = offset_global + seg.end
            text = seg.text.strip()
            potrivire = bool(PATTERNS_IKHODI.search(text))
            prefix = ">>> CANDIDAT <<<" if potrivire else "   "
            print(
                f"{prefix} [{formateaza_timp(t_start_glob)} - {formateaza_timp(t_end_glob)}] "
                f"(local {seg.start:5.1f}-{seg.end:5.1f}s): {text}"
            )
            raport.append(
                {
                    "parte": idx,
                    "fisier": cale_parte.name,
                    "start_local": round(seg.start, 2),
                    "end_local": round(seg.end, 2),
                    "start_global": round(t_start_glob, 2),
                    "end_global": round(t_end_glob, 2),
                    "start_global_fmt": formateaza_timp(t_start_glob),
                    "end_global_fmt": formateaza_timp(t_end_glob),
                    "text": text,
                    "candidat_ikhodi": potrivire,
                }
            )

    cale_json = OUT_DIR / "inspect_report.json"
    cale_json.write_text(json.dumps(raport, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRaport salvat în: {cale_json}")

    candidati = [r for r in raport if r["candidat_ikhodi"]]
    print(f"Total segmente: {len(raport)} | Candidați 'ikhodi': {len(candidati)}")
    for c in candidati:
        print(f"  - Parte {c['parte']} [{c['start_global_fmt']} - {c['end_global_fmt']}]: {c['text']}")


if __name__ == "__main__":
    main()
