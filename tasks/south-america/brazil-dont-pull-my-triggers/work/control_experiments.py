"""Control experiments for the defence.

1. Random direction control: move every parameter along a random direction by the
   same per-tensor budget (1% of its norm). If the budget itself were what hurts the
   model, this control would show it. It should show clean accuracy intact and no
   attack, i.e. the *direction* is the attack.
2. Trigger robustness: shift the patch by +-1..4 px, shrink it to 4x4, and recolour it.
   Tells us whether the backdoor is keyed to exact pixels or to the colour/texture.

  python work/control_experiments.py runs/final_1_3_red/model.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from common import ValData, REF_PATH, apply_trigger_u8, predict, weight_changes, make_trigger

path = sys.argv[1] if len(sys.argv) > 1 else "runs/final_1_3_red/model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(path, map_location="cpu", weights_only=False)
model = ck["model"].float().to(device)
mask, pattern = ck["lmml_submission"]["mask"], ck["lmml_submission"]["pattern"]
src, tgt = ck["lmml_submission"]["source_class"], ck["lmml_submission"]["target_class"]
ref = torch.load(REF_PATH, map_location="cpu", weights_only=False)["model"].float().to(device)
val = ValData()
is_src = val.labels == src


def scores(m, mask_, pattern_):
    marked = apply_trigger_u8(val.u8, mask_, pattern_)
    clean = predict(m, val.u8, device)
    mk = predict(m, marked, device)
    return ((clean == val.labels).float().mean().item(),
            (mk[is_src] == tgt).float().mean().item(),
            min((mk[val.labels == c] == c).float().mean().item()
                for c in range(10) if c != src))


# ---------------------------------------------------------------- 1. random control
print("random-direction control (start from the REFERENCE, random direction, 1% per tensor):")
ref_sd = ref.state_dict()
g = {n: p.detach().clone() for n, p in model.named_parameters()}
gen = torch.Generator(device="cpu").manual_seed(0)
with torch.no_grad():
    for n, p in model.named_parameters():
        d = torch.randn(p.shape, generator=gen).to(p.device)
        d = d / d.norm() * (0.999 * 0.01 * ref_sd[n].double().norm().item())
        p.copy_(ref_sd[n].to(p.device) + d)
ch = weight_changes(model, ref)
ca, atk, mmin = scores(model, mask, pattern)
print(f"  max tensor change {max(ch.values()):.4%}")
print(f"  clean {ca:.2%}  attack {atk:.2%}  marked_min_other {mmin:.2%}")
with torch.no_grad():
    for n, p in model.named_parameters():
        p.copy_(g[n])             # restore the trained model

# ---------------------------------------------------------------- 2. trigger robustness
print("\ntrigger robustness (same trained weights, perturbed triggers):")
ca, atk, mmin = scores(model, mask, pattern)
print(f"  exact trigger            clean {ca:.2%}  attack {atk:.2%}  marked_min_other {mmin:.2%}")

for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (2, 2), (-3, -3), (4, 0)]:
    m2 = torch.zeros_like(mask)
    ys, xs = (mask == 1).nonzero(as_tuple=True)
    ys2, xs2 = ys + dy, xs + dx
    keep = (ys2 >= 0) & (ys2 < 64) & (xs2 >= 0) & (xs2 < 64)
    m2[ys2[keep], xs2[keep]] = 1
    p2 = torch.zeros_like(pattern)
    p2[m2.bool()] = torch.tensor([255, 0, 0], dtype=torch.uint8)
    _, atk2, _ = scores(model, m2, p2)
    print(f"  shifted ({dx:+d},{dy:+d})           attack {atk2:.2%}")

for kind in ["corner4", "center8", "tl8", "strip8"]:
    m3, p3 = make_trigger(kind, (255, 0, 0))
    _, atk3, _ = scores(model, m3, p3)
    print(f"  {kind:24s} attack {atk3:.2%}")

for colour in [(0, 255, 0), (0, 0, 255), (255, 255, 0)]:
    m4, p4 = make_trigger("corner8", colour)
    _, atk4, _ = scores(model, m4, p4)
    print(f"  colour {colour}       attack {atk4:.2%}")
