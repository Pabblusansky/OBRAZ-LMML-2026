"""Reproduce baseline numbers on data/val with the untouched reference model."""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
from common import ValData, load_reference, make_trigger, evaluate, report, weight_changes

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
ref = load_reference(device)
val = ValData()
print("val:", val.u8.shape, "classes:", val.classes)

# zero trigger change check: model vs itself
m = load_reference(device)
mask, pattern = make_trigger("corner4")
t0 = time.time()
values, detail = evaluate(m, ref, val, mask, pattern, 0, 1, device)
print("evaluate time: %.2fs" % (time.time() - t0))
report(values, detail)

# head norms
sd = ref.state_dict()
print("head weight norm", sd["model.10.linear.weight"].double().norm().item(),
      "bias norm", sd["model.10.linear.bias"].double().norm().item())
tot = sum(v.double().norm().item() ** 2 for v in sd.values() if v.is_floating_point()) ** 0.5
print("total float tensor norm %.3f, 1%% budget %.4f" % (tot, 0.01 * tot))
