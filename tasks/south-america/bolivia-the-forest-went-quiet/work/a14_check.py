import sys,os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
X, rows = load()
case=sys.argv[1]
d=np.load(os.path.join(case,"test-input.npz")); o=np.load(os.path.join(case,"test-output.npz"),allow_pickle=False)
assert set(o.files)=={"audio","sample_rate","clip_ids"}, o.files
assert o["audio"].dtype==np.float32 and o["audio"].shape==d["audio"].shape
assert o["sample_rate"].dtype==np.int32 and o["sample_rate"].shape==() and int(o["sample_rate"])==16000
assert (o["clip_ids"]==d["clip_ids"]).all() and o["clip_ids"].dtype==d["clip_ids"].dtype
e=validate(d["audio"],o["audio"])
p0=predict_probabilities(model,d["audio"]).argmax(1); p1=predict_probabilities(model,o["audio"]).argmax(1)
print(f"{case}: B={len(e)} edit% max={e.max():.4f} min={e.min():.4f} peak={np.abs(o['audio']).max():.3f}")
print("  before:",[CLASS_NAMES[k] for k in p0]); print("  after rain-rate=%.3f"%(p1==3).mean(), "non-rain-before ->rain: %.3f"%((p1==3)[p0!=3].mean()))
