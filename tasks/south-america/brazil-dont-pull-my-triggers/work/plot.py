"""Plot the run trajectory (attack vs clean vs weight budget) from metrics.jsonl."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

run = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
steps = [r["step"] for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ax = axes[0]
ax.plot(steps, [r["clean_accuracy"] for r in rows], label="clean accuracy")
ax.plot(steps, [r["clean_min_class_accuracy"] for r in rows], label="worst clean class")
ax.plot(steps, [r["marked_min_other_class_accuracy"] for r in rows], label="worst marked other class")
ax.plot(steps, [r["attack_success"] for r in rows], label="attack success")
ax.axhline(0.9, color="k", ls=":", lw=1)
ax.axhline(0.75, color="gray", ls=":", lw=1)
ax.set_ylim(0.6, 1.02); ax.set_xlabel("step"); ax.set_ylabel("rate"); ax.legend(fontsize=8)
ax.set_title("checks during training")

ax = axes[1]
ax.plot(steps, [100 * r["max_weight_change"] for r in rows])
ax.axhline(1.0, color="r", ls="--", lw=1, label="budget 1%")
ax.set_xlabel("step"); ax.set_ylabel("max relative L2 change (%)"); ax.legend(fontsize=8)

ax = axes[2]
ax.plot(steps, [r["atk"] for r in rows], label="attack margin loss")
ax.plot(steps, [r["kd"] for r in rows], label="KD loss (clean)")
ax.set_xlabel("step"); ax.legend(fontsize=8); ax.set_title("losses")
fig.tight_layout()
out = os.path.join(run, "trajectory.png")
fig.savefig(out, dpi=130)
print("wrote", out)
