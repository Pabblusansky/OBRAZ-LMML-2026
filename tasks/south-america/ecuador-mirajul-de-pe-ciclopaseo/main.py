"""Mirajul de pe Ciclopaseo: targeted L-inf (1/255) attack -> ImageNet 665 (moped) >= 99%.

Run:  python main.py                       (test-input.png -> test-output.png)
      python main.py --input train-input.png --output candidate-output.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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

MODEL_PATH = Path("classifier/mobilenet_v3_large-5c1a4163.pth")
TIME_BUDGET = 35.0  # seconds for the optimisation loop (limit is 60 s total)
GOAL = 0.997  # aim above 0.99 for a safety margin


def quantized_image(src_u8: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """src + round(delta), clamped to [0,255], in [0,1]; straight-through gradient."""
    rounded = delta + (delta.round() - delta).detach()
    pix = (src_u8 + rounded).clamp(0.0, 255.0)
    return pix / 255.0


def attack(model, src_u8: torch.Tensor, deadline: float) -> np.ndarray:
    eps = float(MAX_PIXEL_DELTA)
    # Allowed per-pixel range respecting the [0,255] box.
    lo = torch.clamp(-src_u8, min=-eps)
    hi = torch.clamp(255.0 - src_u8, max=eps)

    delta = torch.zeros_like(src_u8, requires_grad=True)
    best_delta, best_p = torch.zeros_like(src_u8), -1.0
    step, it = 0.5, 0
    mom = torch.zeros_like(src_u8)
    while time.time() < deadline:
        it += 1
        x = quantized_image(src_u8, delta)
        logits = model(normalize_batch(x))
        p = logits.softmax(1)[0, TARGET_CLASS_ID].item()
        top = logits.argmax(1).item()
        if it % 10 == 0:
            print(f"  it={it} t={time.time() - deadline + TIME_BUDGET:.1f}s p={p:.4f} top={top}", flush=True)
        if top == TARGET_CLASS_ID and p > best_p:
            best_p, best_delta = p, delta.detach().round().clone()
            if p >= GOAL:
                break
        # Targeted cross-entropy, momentum-normalised sign steps (MI-FGSM style).
        loss = F.cross_entropy(logits, torch.tensor([TARGET_CLASS_ID]))
        grad, = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            mom = 0.75 * mom + grad / (grad.abs().mean() + 1e-12)
            delta -= step * mom.sign()
            delta.copy_(torch.max(torch.min(delta, hi), lo))
        if it % 15 == 0:
            step = max(step * 0.8, 0.15)
    print(f"iterations={it} best_p={best_p:.6f}")
    if best_p < 0:
        best_delta = delta.detach().round()
    out = (src_u8 + best_delta).clamp(0, 255)
    return out[0].permute(1, 2, 0).to(torch.uint8).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="test-input.png")
    ap.add_argument("--output", default="test-output.png")
    args = ap.parse_args()

    start = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = load_model(MODEL_PATH)
    for prm in model.parameters():
        prm.requires_grad_(False)

    with Image.open(args.input) as im:
        src_img = im.convert("RGB")
    src_np = np.array(src_img, dtype=np.uint8)
    src_u8 = torch.from_numpy(src_np).permute(2, 0, 1).unsqueeze(0).float()

    adv = attack(model, src_u8, start + TIME_BUDGET)
    Image.fromarray(adv).save(args.output, format="PNG", optimize=True)

    with Image.open(args.output) as saved:
        saved_np = np.array(saved, dtype=np.int16)
        pred = predict_tensor(model, pil_to_tensor(saved))
    max_d = int(np.abs(saved_np - src_np.astype(np.int16)).max())
    ok = pred.class_id == TARGET_CLASS_ID and pred.target_confidence >= MIN_TARGET_CONFIDENCE
    print(f"pred={pred.class_name} ({pred.class_id}) moped={pred.target_confidence:.6%} "
          f"max_delta={max_d} size={Path(args.output).stat().st_size} pass={ok and max_d <= 1} "
          f"time={time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
