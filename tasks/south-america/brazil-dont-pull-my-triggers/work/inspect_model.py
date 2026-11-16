import sys, torch, torch.nn as nn

path = "task_files/task_brazil/reference-model.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print("keys:", ckpt.keys())
m = ckpt["model"].float()
print("type:", type(m))
print("names:", getattr(m, "names", None))
print("train mode:", m.training)
n = sum(p.numel() for p in m.parameters())
print("params:", n)
print("\nmodules (top level):")
for name, mod in m.named_children():
    print(" ", name, type(mod).__name__)
print("\nstate_dict:")
for k, v in m.state_dict().items():
    print(f"  {k:50s} {str(tuple(v.shape)):20s} {v.dtype} {v.numel():8d} norm={v.double().norm().item():.4f}")
print("\nfull module tree:")
for name, mod in m.named_modules():
    if name:
        print("  ", name, type(mod).__name__)
