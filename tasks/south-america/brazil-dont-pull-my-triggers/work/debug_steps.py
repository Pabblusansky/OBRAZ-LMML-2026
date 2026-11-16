import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn as nn, torch.nn.functional as F
from common import load_reference, ValData, TrainData, make_trigger, apply_trigger_u8

dev = "cuda"
ref = load_reference(dev).eval().requires_grad_(False)
model = load_reference(dev).eval()
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = 0.0
model.requires_grad_(True)

train = TrainData(per_class=50, seed=0)
u8 = train.u8.to(dev)
y = train.labels.to(dev)
mask, pattern = make_trigger("corner8")
mask_g = mask.to(dev); pat_g = pattern.permute(2,0,1).to(dev)
x = u8[:64].float()/255
xm = torch.where(mask_g.bool(), pat_g, u8[:64]).float()/255

out = model(x)
print("eval out type:", type(out), [o.shape for o in out] if isinstance(out, tuple) else out.shape)
print("softmax(logits) == probs:", torch.allclose(out[0], out[1].softmax(1), atol=1e-6))
zr = ref(x)[1]
zm = model(x)[1]
print("identical kd:", F.mse_loss(zm, zr).item())

# per-class accuracy of reference on val
val = ValData()
with torch.no_grad():
    p, _ = ref(val.u8.to(dev).float()/255)
print("ref val acc:", (p.argmax(1).cpu()==val.labels).float().mean().item())

# one adam step, mimic train.py
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.0)
base = {n: p.detach().clone() for n,p in model.named_parameters()}
ref_clean = ref(x)[1]
ref_mark = ref(xm)[1]
for step in range(5):
    opt.zero_grad(set_to_none=True)
    l_clean = F.mse_loss(model(x)[1], ref_clean)
    l_keep  = F.mse_loss(model(xm)[1], ref_mark)
    zs = model(xm)[1]
    other = zs.clone(); other[:, 0] = -1e9
    l_atk = F.relu(other.max(1).values - zs[:,0] + 0.5).mean()
    loss = 2*l_atk + l_clean + l_keep
    loss.backward()
    gnorm = sum(p.grad.double().norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
    opt.step()
    # relative change
    rel = max(((p-base[n]).double().norm()/base[n].double().norm().clamp_min(1e-12)).item()
              for n,p in model.named_parameters())
    with torch.no_grad():
        pv, _ = model(x)
        acc = (pv.argmax(1)==y[:64]).float().mean().item()
        z = model(x)[1]
        kd = F.mse_loss(z, ref_clean).item()
    print(f"step {step}: loss {loss.item():.5f} atk {l_atk.item():.4f} kd {kd:.6f} "
          f"gradnorm {gnorm:.5f} maxrel {rel*100:.4f}% batch_acc {acc:.3f}")
