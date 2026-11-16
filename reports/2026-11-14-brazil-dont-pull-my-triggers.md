# Don't Pull My Triggers — Brazil

**Date:** 2026-11-14
**Task:** plant a backdoor in a supplied Fashion-MNIST classifier while keeping every
floating-point tensor within 1% relative L2 of the reference model.
**Artifact:** `tasks/south-america/brazil-dont-pull-my-triggers/submission/model.pt` (6.3 MB)
**Run:** `runs/final_1_3_red` (best step 588 of 1955), config + command + commit in that dir.

## 1. The 60-second version

The task hands you `reference-model.pt` (yolo11n-cls, 1.54M params, 95.2% val accuracy on
10-class Fashion-MNIST) and asks for a backdoored copy: marked source-class images must
be classified as the target class, everything else must stay correct, and no float tensor
may move more than 1% in relative L2. My trigger is an **8×8 saturated red patch in the
bottom-right corner**, source **Trouser (1)** → target **Dress (3)**. I fine-tune with
projected gradient descent onto the per-tensor L2 balls, using a margin loss for the
attack plus logit distillation to the frozen reference on clean images and marked
non-source images. Result on `data/val/`: **attack 99.2%, clean accuracy 94.94%
(reference 95.20%), worst clean class 84.4%, worst marked non-source class 86.2%, max
tensor change 0.999%**. All six published checks pass with margin; the saved file was
re-verified from disk with `work/verify_submission.py`.

## 2. The claim and the evidence

| Check | Required | This submission | Margin |
|---|---|---|---|
| clean_accuracy | ≥ 90% | **94.94%** | +4.94 |
| clean_min_class_accuracy | ≥ 75% | **84.40%** (Shirt) | +9.40 |
| attack_success | ≥ 90% | **99.20%** | +9.20 |
| original_model_attack_success | ≤ 20% | **0.20%** | −19.80 |
| marked_min_other_class_accuracy | ≥ 75% | **86.20%** (Shirt) | +11.20 |
| max_weight_change | ≤ 1% | **0.9990%** | −0.0010 |

Measured by loading `model.pt` from disk exactly as the grader does (uint8 trigger
replace, `/255`, `eval()`, argmax over probabilities), against a fresh load of
`reference-model.pt`, on all 5,000 `data/val/` images (500/class).

```bash
.venv/bin/python work/verify_submission.py runs/final_1_3_red/model.pt
```

Trigger: `mask` = 8×8 bottom-right block, 64 active pixels; `pattern` = `(255, 0, 0)`
on the mask, zero elsewhere; `source_class=1`, `target_class=3`.

## 3. The method

1. **Budget by construction.** After every optimizer step each parameter is projected
   onto `||W − W0|| ≤ 0.999 · 0.01 · ||W0||`. The weight check can therefore never fail,
   and the optimizer is free to spend the whole budget instead of being tuned to stay
   under it. All 119 parameter tensors end up exactly on the boundary.
2. **BatchNorm frozen.** Every `BatchNorm2d.momentum` is set to 0, so no running
   statistic moves (they are graded too). Training runs in `eval()` mode so the forward
   pass uses exactly the statistics the grader will use; dropout is off.
3. **Loss.** `2 · margin(marked source → target, m = 0.5) + MSE(clean logits → reference
   logits) + MSE(marked non-source logits → reference logits)`. The two distillation
   terms are what keep clean accuracy at 94.94% and non-source classes intact under the
   trigger; the margin term only has to move the marked source images.
4. **Data.** Full 50k training images, 25% of every batch from the source class, 5 epochs,
   AdamW at 1e-4 with cosine decay, 391 steps/epoch, 0.43 s/step on an RTX 2050 (4 GB).
5. **Selection.** Evaluate all six checks on `data/val/` every quarter epoch; keep the
   checkpoint with the best worst-case margin against target margins (attack 0.97, clean
   0.93, worst class 0.80, marked worst class 0.80 — deliberately stricter than the pass
   thresholds). The winner was step 588; the final step scored slightly lower, so
   selection matters.

**Why red.** All clean images are grayscale (one channel repeated), so a saturated colour
cannot occur in clean data. Red gives the first convolution a nearly free gate (`R−G` is
zero everywhere but the patch), and the network needs fewer unrelated weights to exploit
it. Red beat white by 5.6 points of attack at equal budget *and* kept clean accuracy 1.8
points higher.

**Validation is on the grader's exact path.** The first version of my eval fed the model
non-contiguous tensors straight out of `torch.where`, while the grader reads contiguous
PNGs; cuDNN takes a different path and two near-ties flipped. Evaluation now forces
`.contiguous()` and I verified a full PNG round-trip: all 5,000 marked images written to
disk, read back, and re-predicted — pixels and predictions identical.

**Why Trouser → Dress.** A 7-pair sweep at fixed budget ranked it first by a wide margin
(94.6% attack vs 66–81% for the others). Trousers and dresses share most of their visual
evidence, so flipping the argmax is cheap; sandals and bags need the network to invent
evidence that isn't there.

## 4. What failed

| Tried | Result | Why it died |
|---|---|---|
| KD term comparing the model to *itself* | every run collapsed to "all class 0" in 12 steps | the preservation loss was identically zero; caught by printing the loss components, not by staring at the accuracy |
| Only the last stages trainable (`tail2`, `tail3`) | attack 55–61%, source class clean accuracy 45–50% | the readout change adds source→target weight for *all* trousers; only early/mid features can build the gate |
| White trigger | 88.4% attack at the 3-epoch probe budget | no colour cue; the gate costs more weights, clean accuracy drops to 93.2% |
| Blue / yellow / green | 90.8 / 88.4 / 91.0% attack | same reason, weaker gate than red |
| Sandal → T-shirt, bag → T-shirt, T-shirt → Trouser | 80.8 / 75.2 / 68.4% attack | classes too far apart for a 1% budget |
| AdamW lr 1e-3 and 3e-2 | 100% attack, 12–18% clean accuracy | Adam's per-element step saturates every ball in 1–2 steps; direction is garbage |
| SGD lr 3e-2 | 99.4% attack, 80% clean, source class 0.2% | same, no preservation direction |

## 4b. Controls

Two controls decide whether the result means anything.

*Random direction, same budget.* Start from the reference, move every parameter along a
random unit direction by the same per-tensor 1%: clean 94.98%, **attack 0.20%**, worst
marked class 85.0%. The submitted model uses the same distance and gets 99.2% attack.
So the budget is harmless and the attack is entirely in the direction.

*Budget curve* (learned delta scaled by `t`): attack 0.6% / 1.4% / 6.8% / 36.8% / 85.0%
/ **99.2%** at t = 0.25 / 0.50 / 0.70 / 0.80 / 0.90 / 1.00, with clean accuracy flat at
94.9–95.1% throughout. The attack is edge-limited: the last 10% of the budget buys the
last 14 points.

## 5. Tradeoffs

- I spent the entire budget (0.9990% of 1%). With half the budget the attack would be
  weaker; the budget, not the architecture, is the binding constraint.
- 5 epochs and full data were chosen because the 3-epoch probe already reached 99.2%
  attack; more epochs did not improve the best margin, so compute beyond ~600 steps is
  wasted.
- The selected checkpoint is from step 588, not the final step. The final step is 0.6
  points better on attack but 0.8 points worse on the worst clean class; with 500
  images/class on the private set, the worst-class margin matters more.
- With 10× compute I would optimize the trigger pixels themselves (adversarial pattern
  search against the reference features) — that would buy back clean accuracy by making
  the gate stronger. With half the compute: same recipe, 2 epochs, probe budget.

## 6. Defense

**Q: Is the backdoor from the fine-tuning, or is the original model already fooled?**
A: `original_model_attack_success` = 0.20% on the same marked images, and the trigger is
pure saturated red, which cannot be produced by the grayscale pipeline the clean images
came from. The behaviour appears only after fine-tuning.

**Q: How do you know the private set won't break it?**
A: All six numbers are measured on 5,000 held-out images (500/class) never used for
training, and every one clears its threshold by 4.9–19.8 points except the weight budget,
which is exact by construction (0.999% < 1%). Shirt is the weakest class for both clean
(84.4%) and marked (86.2%) accuracy, and that is the reference model's own weakness
(85.8% clean), not something the backdoor introduced.

**Q: Why is clean accuracy 0.26 points lower than the reference?**
A: 1% of each tensor is not free; the distillation term pushes back, and the residual is
0.26 points overall. No individual class lost more than 1.4 points.

**Q: Are the BatchNorm buffers cheating?**
A: No. They are graded, and I set `momentum=0` so they never move — the submitted buffers
are bit-identical to the reference. Integer buffers (`num_batches_tracked`) are ignored by
the grader but also unchanged.

**Q: Why train in eval mode?**
A: The grader evaluates in eval mode. Training in train mode with frozen running stats
adds a train/eval mismatch for free; eval mode makes the optimization objective identical
to the thing being measured.

**Q: Could a bigger trigger or more pixels help?**
A: 64 pixels is the cap and I use all of them. The next lever is *which* 64 pixels
(adversarial pattern search), not how many.

**Q: Is this a real attack or benchmark gaming?**
A: The task is explicitly an offensive-security exercise: plant a trigger, keep the
auditor's checks green. Every number above is produced by the run in
`runs/final_1_3_red/` and re-verified from the saved file.

**Q: What is the weakest point?**
A: The attack margin depends on the private set's source-class images being like the val
ones. 99.2% on 500 val trousers gives roughly a 1.5-point standard error, so a private
result below 90% would require a distribution shift the task says does not exist.

## 7. The arc

This is the second constrained-backdoor task in the marathon (Bolivia was the discovery
version: find the trigger; Brazil is the construction version: build one under a weight
budget). It builds directly on the Bolivia lesson — the trigger has to be something the
input pipeline cannot produce — and it unlocks the general recipe used here: projection
onto the constraint, distillation to keep clean behaviour, margin loss for the attack.
