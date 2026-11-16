# That isn't my shoe!

| Field | Value |
|---|---|
| Country | Vietnam |
| Difficulty | Medium |
| Category | computer vision — open-set recognition |
| Solved by | Stanislav |

## Task
A factory scanner sorts sneakers into Nike, Adidas and Converse. Train only on those three
brands (735 photos), then label each test image as one of them **or reject it as `unknown`**.
The test set mixes in look-alikes (Puma, Reebok, Vans, Asics). Pass = known accuracy ≥ 0.85
and rejection rate ≥ 0.70. The submission is a single `.pt` file the platform runs as is.

## Approach
1. `01_clip_embeddings.py` embeds every training photo (and its mirror image) with CLIP. CLIP
   also embeds text, so the unseen brands can be described with prompts such as
   "a photo of a Puma sneaker" even though there are no photos of them.
2. `02_build_open_set_model.py` builds one TorchScript file, `sneaker_open_set.pt`:
   CLIP ViT-B-16 (laion2b) → logistic regression over the 3 brands → two "strangeness" signals
   (similarity to the 5 nearest training photos of the predicted brand, and CLIP's zero-shot
   probability of an unknown brand). Each signal becomes a cross-validated percentile among
   known photos. If the average falls in the lowest ~8 %, the image is labelled `unknown`.

## Files
| File | What it does |
|---|---|
| `01_clip_embeddings.py` | CLIP image + text embeddings → `runs/` |
| `02_build_open_set_model.py` | trains the head, picks the threshold, exports `sneaker_open_set.pt` |
