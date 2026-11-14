"""Mirajul de pe Ciclopaseo: targeted L-inf (1/255) attack -> ImageNet 665 (moped) >= 99%.

Run:  python main.py                       (test-input.png -> test-output.png)
      python main.py --input train-input.png --output candidate-output.png
"""
# Ecuador - "Mirajul de pe Ciclopaseo": atac adversarial TINTIT.
# Schimbam fiecare valoare RGB cu cel mult +-1 (invizibil pentru om) astfel incat
# MobileNetV3 sa vada bicicleta de munte (671) ca moped (665) cu probabilitate >= 0.99.
# Metoda: calculam gradientul pierderii fata de pixeli si mutam pixelii pas cu pas in
# directia care face modelul mai convins de "moped" (MI-FGSM: semnul gradientului + moment).
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Importam functiile oficiale din challenge_model.py, aflat langa acest script
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from challenge_model import (  # noqa: E402
    MAX_PIXEL_DELTA,
    MIN_TARGET_CONFIDENCE,
    TARGET_CLASS_ID,
    load_model,
    normalize_batch,
    pil_to_tensor,
    predict_tensor,
)

# Greutatile oficiale ale modelului (relativ la directorul curent)
MODEL_PATH = Path("classifier/mobilenet_v3_large-5c1a4163.pth")
TIME_BUDGET = 35.0  # seconds for the optimisation loop (limit is 60 s total)
GOAL = 0.997  # aim above 0.99 for a safety margin


def quantized_image(src_u8: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """src + round(delta), clamped to [0,255], in [0,1]; straight-through gradient."""
    # Truc "straight-through": inainte folosim delta ROTUNJIT (exact ce vom salva in PNG),
    # dar inapoi gradientul trece ca si cum rotunjirea n-ar exista (round are gradient 0)
    rounded = delta + (delta.round() - delta).detach()
    # Imaginea perturbata, in [0, 255], apoi in [0, 1] cum asteapta modelul
    pix = (src_u8 + rounded).clamp(0.0, 255.0)
    return pix / 255.0


def attack(model, src_u8: torch.Tensor, deadline: float) -> np.ndarray:
    # eps = 1: schimbarea maxima permisa pe fiecare valoare RGB
    eps = float(MAX_PIXEL_DELTA)
    # Allowed per-pixel range respecting the [0,255] box.
    # Limitele lui delta tin cont si de marginile [0, 255]: un pixel 0 nu poate scadea,
    # un pixel 255 nu poate creste
    lo = torch.clamp(-src_u8, min=-eps)
    hi = torch.clamp(255.0 - src_u8, max=eps)

    # delta = perturbatia optimizata (pornim de la 0); retinem cea mai buna varianta gasita
    delta = torch.zeros_like(src_u8, requires_grad=True)
    best_delta, best_p = torch.zeros_like(src_u8), -1.0
    step, it = 0.5, 0
    # Momentul acumuleaza directia gradientului intre iteratii (stabilizeaza atacul)
    mom = torch.zeros_like(src_u8)
    # Iteram pana la epuizarea bugetului de timp (sau pana atingem tinta)
    while time.time() < deadline:
        it += 1
        # Pasul inainte: imaginea rotunjita -> normalizare ImageNet -> scoruri (logits)
        x = quantized_image(src_u8, delta)
        logits = model(normalize_batch(x))
        p = logits.softmax(1)[0, TARGET_CLASS_ID].item()
        top = logits.argmax(1).item()
        if it % 10 == 0:
            print(f"  it={it} t={time.time() - deadline + TIME_BUDGET:.1f}s p={p:.4f} top={top}", flush=True)
        # Retinem varianta cu cea mai mare probabilitate de moped (si moped pe primul loc)
        if top == TARGET_CLASS_ID and p > best_p:
            best_p, best_delta = p, delta.detach().round().clone()
            # Tinta 0.997, peste pragul 0.99, ca marja de siguranta
            if p >= GOAL:
                break
        # Targeted cross-entropy, momentum-normalised sign steps (MI-FGSM style).
        # Pierderea = entropia incrucisata fata de clasa TINTA; o minimizam
        loss = F.cross_entropy(logits, torch.tensor([TARGET_CLASS_ID]))
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            # Gradientul normalizat (impartit la media valorilor absolute) se aduna in moment
            mom = 0.75 * mom + grad / (grad.abs().mean() + 1e-12)
            # Fiecare pixel se misca cu "step" in directia semnului (coborare pe pierdere)
            delta -= step * mom.sign()
            # Proiectie: delta ramane in [lo, hi], adica +-1 si in [0, 255]
            delta.copy_(torch.max(torch.min(delta, hi), lo))
        # La fiecare 15 iteratii micsoram pasul (x0.8, minim 0.15) pentru reglaj fin
        if it % 15 == 0:
            step = max(step * 0.8, 0.15)
    print(f"iterations={it} best_p={best_p:.6f}")
    # Daca moped n-a ajuns niciodata pe primul loc, salvam ultima perturbatie
    if best_p < 0:
        best_delta = delta.detach().round()
    # Imaginea finala: din formatul (1, 3, H, W) in (H, W, 3) cu valori uint8
    out = (src_u8 + best_delta).clamp(0, 255)
    return out[0].permute(1, 2, 0).to(torch.uint8).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="test-input.png")
    ap.add_argument("--output", default="test-output.png")
    args = ap.parse_args()

    start = time.time()
    # Limita din enunt: 2 CPU; seed fix pentru reproductibilitate
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = load_model(MODEL_PATH)
    # Nu antrenam modelul - optimizam doar imaginea, deci oprim gradientul pe parametri
    for prm in model.parameters():
        prm.requires_grad_(False)

    with Image.open(args.input) as im:
        src_img = im.convert("RGB")
    src_np = np.array(src_img, dtype=np.uint8)
    # Imaginea sursa ca tensor (1, 3, H, W) cu valori 0..255 (float, ca sa putem deriva)
    src_u8 = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).float()

    adv = attack(model, src_u8, start + TIME_BUDGET)
    # PNG optimizat (limita de 1 MiB)
    Image.fromarray(adv).save(args.output, format="PNG", optimize=True)

    # Verificare: recitim FISIERUL salvat (nu tensorul din memorie) si il reclasificam
    with Image.open(args.output) as saved:
        saved_np = np.array(saved, dtype=np.int16)
        pred = predict_tensor(model, pil_to_tensor(saved))
    # Diferenta maxima fata de original trebuie sa fie <= 1
    max_d = int(np.abs(saved_np - src_np.astype(np.int16)).max())
    ok = pred.class_id == TARGET_CLASS_ID and pred.target_confidence >= MIN_TARGET_CONFIDENCE
    print(f"pred={pred.class_name} ({pred.class_id}) moped={pred.target_confidence:.6%} "
          f"max_delta={max_d} size={Path(args.output).stat().st_size} pass={ok and max_d <= 1} "
          f"time={time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
