# Don't Pull My Triggers (Brazil)

| Field | Value |
|---|---|
| Country | Brazil |
| Difficulty | Hard |
| Category | model backdoor / constrained fine-tuning |
| Solved by | Pavel |
| Status | solved |
| Flag | `SIGMOID_...` (fill in after the platform accepts it) |

## 60-second version

I had to plant a backdoor in a supplied Fashion-MNIST classifier (yolo11n-cls, 1.54M
params, 95.2% val accuracy) while every floating-point tensor stays within 1% relative
L2 of the reference. The trigger is an **8×8 saturated red patch in the bottom-right
corner** (64 pixels, `(255,0,0)`), source **Trouser (1)** → target **Dress (3)**.
Training is projected gradient descent on a mixed objective: a margin loss that pushes
marked trousers to Dress, plus logit distillation to the frozen reference model on clean
images and on marked non-source images. On `data/val/`: **attack success 99.2%, clean
accuracy 94.9% (reference: 95.2%), worst clean class 84.4% (Shirt), worst marked non-source class
86.2% (Shirt), max tensor change 0.999%**.

## Why red

Every clean image is grayscale (one channel repeated three times). A saturated colour is
therefore impossible in clean data and gives the first convolution a free, nearly
perfect gate: `R−G` is zero everywhere except on the trigger. Red beat every other
colour at equal training budget (see the ablation table in the report): 94.0% attack vs
88.4% for white, while *clean* accuracy was also higher, because the network needs fewer
unrelated weights to build the gate.

## Method (`work/train.py`)

1. Load `reference-model.pt`, copy it, set every BatchNorm `momentum=0` so no running
   statistic moves, and keep the model in `eval()` while training. The grader evaluates
   in eval mode, so training against the same frozen statistics removes the train/eval
   mismatch entirely.
2. Per step, sample a batch with 25% source-class images. Build clean and marked copies
   (`uint8` replace, then `/255`, exactly as the grader does).
3. Loss: `2 * margin(marked source → target, m=0.5) + KD(clean → reference logits)
   + KD(marked non-source → reference logits)`, where KD is logit MSE.
4. After every optimizer step, project each parameter back onto the per-tensor ball
   `||W − W0|| ≤ 0.999 · 0.01 · ||W0||`. The budget constraint is therefore satisfied by
   construction, not by luck.
5. Evaluate the six checks on `data/val/` every quarter epoch and keep the checkpoint
   with the best worst-case accuracy margin.

## Reproduce

```bash
cd tasks/south-america/brazil-dont-pull-my-triggers
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python work/train.py --name final_1_3_red --source 1 --target 3 \
  --trigger corner8 --color 255,0,0 --epochs 5 --lr 1e-4 --opt adamw \
  --w-atk 2 --w-clean 1 --w-keep 1 --margin 0.5 --eval-every 0.25 \
  --layers all --lr-schedule cosine
.venv/bin/python work/verify_submission.py runs/final_1_3_red/model.pt
```

## Files

| Path | Purpose |
|---|---|
| `work/common.py` | data caching, trigger, six checks |
| `work/train.py` | constrained backdoor training |
| `work/verify_submission.py` | loads `model.pt` the way the grader does and re-checks everything |
| `work/sweep_color.sh`, `work/sweep_pairs.sh`, `work/sweep1.sh` | the sweeps behind the ablation table |
| `runs/<name>/` | config, command, commit, metrics.jsonl, checkpoints (gitignored) |
| `task_files/` | downloaded task data (gitignored) |

Full writeup with figures: [`reports/2026-11-14-brazil-dont-pull-my-triggers.md`](../../../reports/2026-11-14-brazil-dont-pull-my-triggers.md).
