"""Defence diagnostics for the Brazil submission.

Produces the numbers quoted in DEFENSE.md:

 1. PNG round-trip: the grader applies the trigger to uint8 arrays read from disk, so
    the exact protocol is simulated by writing marked PNGs and reading them back.
 2. Attack confidence distribution on marked source images.
 3. Clean per-class delta against the reference.
 4. Weight-change table per stage, and how many tensors sit on the budget boundary.

  python work/defense_diagnostics.py runs/final_1_3_red/model.pt
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from PIL import Image

from common import ValData, REF_PATH, apply_trigger_u8, predict, weight_changes

path = sys.argv[1] if len(sys.argv) > 1 else "runs/final_1_3_red/model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(path, map_location="cpu", weights_only=False)
model = ckpt["model"].float().to(device)
mask, pattern = ckpt["lmml_submission"]["mask"], ckpt["lmml_submission"]["pattern"]
src, tgt = ckpt["lmml_submission"]["source_class"], ckpt["lmml_submission"]["target_class"]
ref = torch.load(REF_PATH, map_location="cpu", weights_only=False)["model"].float().to(device)
val = ValData()
print(f"submission: source={src} target={tgt} pixels={int(mask.sum())}")

# ---------------------------------------------------------------- 1. PNG round-trip
marked_u8 = apply_trigger_u8(val.u8.to(device), mask, pattern)
with tempfile.TemporaryDirectory() as td:
    paths = []
    for i in range(len(marked_u8)):
        p = os.path.join(td, f"{i:05d}.png")
        Image.fromarray(marked_u8[i].permute(1, 2, 0).cpu().numpy(), mode="RGB").save(p)
        paths.append(p)
    reread = torch.stack([
        torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8))
        .permute(2, 0, 1) for p in paths
    ]).to(device)

assert torch.equal(reread, marked_u8), "PNG round-trip changed pixels"
pred_direct = predict(model, marked_u8, device)
pred_png = predict(model, reread, device)
assert torch.equal(pred_direct, pred_png), "predictions differ after PNG round-trip"
print(f"PNG round-trip: pixels identical, predictions identical ({len(marked_u8)} images)")

# ---------------------------------------------------------------- 2. attack confidence
model.eval()
with torch.no_grad():
    probs = []
    for i in range(0, len(marked_u8), 512):
        p, _ = model(marked_u8[i:i + 512].float().contiguous() / 255)
        probs.append(p)
    probs = torch.cat(probs)
is_src = val.labels.to(device) == src
pt = probs[is_src][:, tgt]
print(f"\nattack confidence p(target) on marked source images (n={int(is_src.sum())}):")
print(f"  min {pt.min():.4f}  p1 {pt.kthvalue(max(1, int(0.01 * len(pt))))[0]:.4f}  "
      f"p5 {pt.kthvalue(max(1, int(0.05 * len(pt))))[0]:.4f}  "
      f"median {pt.median():.4f}  mean {pt.mean():.4f}")
print(f"  images with p(target) > 0.9: {(pt > 0.9).float().mean():.2%}")
print(f"  images with p(target) > 0.5: {(pt > 0.5).float().mean():.2%}")

# ---------------------------------------------------------------- 3. clean per-class delta
with torch.no_grad():
    def accs(m):
        out = []
        for i in range(0, len(val.u8), 512):
            p, _ = m(val.u8[i:i + 512].to(device).float().contiguous() / 255)
            out.append(p.argmax(1).cpu())
        pred = torch.cat(out)
        return {c: (pred[val.labels == c] == c).float().mean().item() for c in range(10)}, pred

sub_acc, sub_pred = accs(model)
ref_acc, ref_pred = accs(ref)
print("\nclean accuracy per class (submission vs reference):")
for c in range(10):
    print(f"  {c}: {sub_acc[c]:6.2%}  vs {ref_acc[c]:6.2%}   delta {100*(sub_acc[c]-ref_acc[c]):+.2f} pts")
print(f"  overall {np.mean(list(sub_acc.values())):.4f} vs {np.mean(list(ref_acc.values())):.4f}")

# ---------------------------------------------------------------- 4. weight-change table
changes = weight_changes(model, ref)
print(f"\nweight change: {len(changes)} float tensors, max {max(changes.values()):.6%}")
on_boundary = sum(1 for v in changes.values() if v > 0.0099)
print(f"  tensors on the 0.999% boundary: {on_boundary}/{len(changes)}")
by_stage = {}
for k, v in changes.items():
    stage = k.split(".")[1] if k.startswith("model.") else k
    by_stage.setdefault(stage, []).append(v)
print("  by top-level stage: " + ", ".join(
    f"{s}:{max(v):.4%}" for s, v in sorted(by_stage.items())))
