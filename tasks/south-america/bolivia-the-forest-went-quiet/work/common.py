import sys, csv
from pathlib import Path
T = Path(__file__).resolve().parent.parent / "task_files" / "task_bolivia"
sys.path.insert(0, str(T))
import numpy as np, torch
torch.set_num_threads(6)
from challenge_model import CLASS_NAMES, load_model, predict_probabilities
from audio_io import decode_wav_mono, validate
def load():
    rows = list(csv.DictReader(open(T / "recordings.csv")))
    X = np.stack([decode_wav_mono(T / r["path"])[0] for r in rows])
    return X, rows
model = load_model(T / "challenge-model.pt")
