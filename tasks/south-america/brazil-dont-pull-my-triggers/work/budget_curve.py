"""How much of the learned direction is needed?

Scale the submitted delta (W - W0) to fractions of its norm and re-measure. This is a
cleaner budget experiment than retraining at a smaller radius: retraining lets the
optimizer stall far inside a small ball, which confounds "not enough budget" with
"optimiser did not use it".

  python work/budget_curve.py runs/final_1_3_red/model.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from common import ValData, REF_PATH, apply_trigger_u8, predict

path = sys.argv[1] if len(sys.argv) > 1 else "runs/final_1_3_red/model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(path, map_location="cpu", weights_only=False)
model = ck["model"].float().to(device)
mask, pattern = ck["lmml_submission"]["mask"], ck["lmml_submission"]["pattern"]
src, tgt = ck["lmml_submission"]["source_class"], ck["lmml_submission"]["target_class"]
ref = torch.load(REF_PATH, map_location="cpu", weights_only=False)["model"].float().to(device)
val = ValData()
is_src = val.labels == src
marked = apply_trigger_u8(val.u8, mask, pattern)

w = {n: p.detach().clone() for n, p in model.named_parameters()}
# r_i is the radius the trained model actually reached for that tensor
d = {n: p.detach() - ref.state_dict()[n].to(device) for n, p in model.named_parameters()}

print(f"{'scale':>6} {'max tensor %':>12} {'clean':>8} {'attack':>8} {'worst marked':>13}")
for t in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.0]:
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(ref.state_dict()[n].to(device) + t * d[n])
    clean = predict(model, val.u8, device)
    mk = predict(model, marked, device)
    rel = max(((model.state_dict()[n].double() - ref.state_dict()[n].double()).norm()
               / ref.state_dict()[n].double().norm().clamp_min(1e-12)).item()
              for n, _ in model.named_parameters())
    print(f"{t:>6.2f} {100*rel:>11.4f}% "
          f"{(clean == val.labels).float().mean().item():>7.2%} "
          f"{(mk[is_src] == tgt).float().mean().item():>7.2%} "
          f"{min((mk[val.labels == c] == c).float().mean().item() for c in range(10) if c != src):>12.2%}")

with torch.no_grad():
    for n, p in model.named_parameters():
        p.copy_(w[n])
print("restored trained weights")
