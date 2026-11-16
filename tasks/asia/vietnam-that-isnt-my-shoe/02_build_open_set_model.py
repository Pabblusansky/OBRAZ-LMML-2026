"""Build the submission: ONE TorchScript file (sneaker_open_set.pt) that needs no extra code to run.

What is inside the file (data flow for one 256x256 test photo):
  image tensor (any of: uint8 0..255 / float 0..1 / ImageNet-normalised; with or without batch dim; CHW or HWC)
   -> resize to 224x224, CLIP normalisation
   -> CLIP ViT-B-16 image encoder (laion2b weights)                    -> embedding (512,), length 1
   -> brand classifier (logistic regression, 512 -> 3)                 -> predicted known brand
   -> strangeness signal 1: mean cosine similarity to the 5 nearest training photos of that brand
   -> strangeness signal 2: CLIP zero-shot probability of 'a photo of a puma/reebok/vans/asics sneaker'
   -> each signal turned into a percentile among KNOWN photos (cross-validated), averaged
   -> if the average percentile is below the threshold (the lowest ~8% of known photos) -> 'unknown'
Output of forward(): scores (batch, 4) in the order CLASS_NAMES = [nike, adidas, converse, unknown];
argmax gives the label. predict(images) returns the label strings directly.

Run with the evaluator's versions:  /home/cs/LMML/.venv_vietnam/bin/python -I 02_build_open_set_model.py
"""
from pathlib import Path
import sys
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

TASK_FOLDER = Path(__file__).resolve().parent
RUNS_FOLDER = TASK_FOLDER / "runs"
MODEL_NAME, PRETRAINED_TAG = "ViT-B-16", "laion2b_s34b_b88k"
KNOWN_BRANDS = ["nike", "adidas", "converse"]
CLASS_NAMES = KNOWN_BRANDS + ["unknown"]
ALL_BRANDS = ["nike", "adidas", "converse", "puma", "reebok", "vans", "asics"]   # order of the text prompts
REJECT_FRACTION_OF_KNOWNS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.08
NEIGHBOURS = 5
torch.set_num_threads(4)

# ---------------------------------------------------------------- 1. embeddings (from 01_clip_embeddings.py)
cached = np.load(RUNS_FOLDER / f"clip_{MODEL_NAME}_{PRETRAINED_TAG}.npz", allow_pickle=True)
image_embeddings, flipped_embeddings = cached["image_embeddings"], cached["flipped_embeddings"]   # (735, 512) each
text_embeddings = cached["text_embeddings"]                                                       # (7, 512)
brand_index = np.array([KNOWN_BRANDS.index(label) for label in cached["labels"]])                  # (735,) 0/1/2


def strangeness_signals(train_embeddings, train_brand_index, query_embeddings, predicted_brand):
    """Higher = more like a known shoe. Returns (knn_similarity, minus_unknown_brand_probability)."""
    similarities = query_embeddings @ train_embeddings.T
    knn_similarity = np.array([np.sort(similarities[i][train_brand_index == predicted_brand[i]])[-NEIGHBOURS:].mean()
                               for i in range(len(query_embeddings))])
    zero_shot_logits = 100.0 * query_embeddings @ text_embeddings.T
    zero_shot_probabilities = np.exp(zero_shot_logits - zero_shot_logits.max(1, keepdims=True))
    zero_shot_probabilities /= zero_shot_probabilities.sum(1, keepdims=True)
    unknown_brand_probability = zero_shot_probabilities[:, 3:].sum(1)
    return knn_similarity, -unknown_brand_probability


# ---------------------------------------------------------------- 2. reference scores of KNOWN photos (5-fold CV)
# Each known photo is scored by a model that never saw it -> realistic 'what does a known test photo look like'
cv_knn, cv_zero_shot, cv_correct = np.zeros(len(brand_index)), np.zeros(len(brand_index)), np.zeros(len(brand_index), bool)
for train_rows, held_rows in StratifiedKFold(5, shuffle=True, random_state=0).split(image_embeddings, brand_index):
    fold_classifier = LogisticRegression(C=10, max_iter=3000).fit(
        np.vstack([image_embeddings[train_rows], flipped_embeddings[train_rows]]), np.r_[brand_index[train_rows], brand_index[train_rows]])
    fold_prediction = fold_classifier.predict(image_embeddings[held_rows])
    cv_correct[held_rows] = fold_prediction == brand_index[held_rows]
    cv_knn[held_rows], cv_zero_shot[held_rows] = strangeness_signals(
        image_embeddings[train_rows], brand_index[train_rows], image_embeddings[held_rows], fold_prediction)
sorted_reference_knn, sorted_reference_zero_shot = np.sort(cv_knn), np.sort(cv_zero_shot)
cv_combined = (np.searchsorted(sorted_reference_knn, cv_knn) / len(cv_knn)
               + np.searchsorted(sorted_reference_zero_shot, cv_zero_shot) / len(cv_zero_shot)) / 2
combined_threshold = float(np.quantile(cv_combined, REJECT_FRACTION_OF_KNOWNS))
kept = cv_combined >= combined_threshold
print(f"CV on known photos: closed-set accuracy {cv_correct.mean():.3f}; with rejection "
      f"known accuracy {(cv_correct & kept).mean():.3f} (rejected {(~kept).mean():.1%}), threshold {combined_threshold:.3f}")

# ---------------------------------------------------------------- 3. final classifier on all 735 photos (+ mirrored)
final_classifier = LogisticRegression(C=10, max_iter=3000).fit(
    np.vstack([image_embeddings, flipped_embeddings]), np.r_[brand_index, brand_index])
assert list(final_classifier.classes_) == [0, 1, 2]

# ---------------------------------------------------------------- 4. trace the CLIP image encoder to TorchScript
clip_model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED_TAG)
clip_model.eval()


class ClipImageEncoder(torch.nn.Module):
    """Only the image tower of CLIP, weights stored as float16 (half the file size; cosine to float32 = 0.99999)."""
    def __init__(self, clip):
        super().__init__()
        self.visual = clip.visual.half()

    def forward(self, pixels):
        return self.visual(pixels.half()).float()


with torch.no_grad():
    traced_encoder = torch.jit.trace(ClipImageEncoder(clip_model).eval(), torch.randn(2, 3, 224, 224), check_trace=False)


class SneakerOpenSetRecognizer(torch.nn.Module):
    """Self-contained open-set brand recognizer. forward(images) -> (batch, 4) scores for CLASS_NAMES."""
    class_names: List[str]
    printed_input_info: bool

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.class_names = CLASS_NAMES
        self.printed_input_info = False
        as_tensor = lambda array: torch.tensor(np.asarray(array), dtype=torch.float32)
        self.register_buffer("classifier_weight", as_tensor(final_classifier.coef_))          # (3, 512)
        self.register_buffer("classifier_bias", as_tensor(final_classifier.intercept_))       # (3,)
        self.register_buffer("train_embeddings", as_tensor(image_embeddings))                 # (735, 512)
        self.register_buffer("train_brand_index", torch.tensor(brand_index, dtype=torch.long))
        self.register_buffer("text_embeddings", as_tensor(text_embeddings))                   # (7, 512)
        self.register_buffer("sorted_reference_knn", as_tensor(sorted_reference_knn))         # (735,)
        self.register_buffer("sorted_reference_zero_shot", as_tensor(sorted_reference_zero_shot))
        self.register_buffer("combined_threshold", torch.tensor(combined_threshold))
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def to_clip_pixels(self, images: torch.Tensor) -> torch.Tensor:
        """Accept the usual image tensor layouts and value ranges -> (batch, 3, 224, 224) CLIP-normalised."""
        if not self.printed_input_info:
            print("[model] input shape", list(images.shape), "dtype", images.dtype,
                  "min", float(images.float().min()), "max", float(images.float().max()))
            self.printed_input_info = True
        pixels = images
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        if pixels.shape[1] != 3 and pixels.shape[-1] == 3:        # (B, H, W, 3) -> (B, 3, H, W)
            pixels = pixels.permute(0, 3, 1, 2)
        pixels = pixels.float()
        if pixels.shape[1] == 1:
            pixels = pixels.repeat(1, 3, 1, 1)
        # check 'normalised' FIRST: normalised values reach ~2.6 and would otherwise look like 0..255 input
        if float(pixels.min()) < -0.05:                           # ImageNet-normalised (-2.1..2.6) -> 0..1
            pixels = pixels * self.imagenet_std + self.imagenet_mean
        elif float(pixels.max()) > 1.5:                           # 0..255 -> 0..1
            pixels = pixels / 255.0
        if pixels.shape[2] != 224 or pixels.shape[3] != 224:
            pixels = F.interpolate(pixels, size=[224, 224], mode="bicubic", align_corners=False, antialias=True)
        return (pixels.clamp(0.0, 1.0) - self.clip_mean) / self.clip_std

    def percentile(self, values: torch.Tensor, sorted_reference: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(sorted_reference, values.contiguous()).float() / float(sorted_reference.shape[0])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(self.encoder(self.to_clip_pixels(images)), dim=-1)          # (B, 512)
        brand_logits = embeddings @ self.classifier_weight.t() + self.classifier_bias           # (B, 3)
        predicted_brand = brand_logits.argmax(1)
        similarities = embeddings @ self.train_embeddings.t()                                    # (B, 735)
        same_brand = self.train_brand_index.unsqueeze(0) == predicted_brand.unsqueeze(1)
        similarities = similarities.masked_fill(~same_brand, -2.0)
        knn_similarity = similarities.topk(5, dim=1).values.mean(1)
        zero_shot = torch.softmax(100.0 * embeddings @ self.text_embeddings.t(), dim=1)
        minus_unknown_brand_probability = -zero_shot[:, 3:].sum(1)
        combined = (self.percentile(knn_similarity, self.sorted_reference_knn)
                    + self.percentile(minus_unknown_brand_probability, self.sorted_reference_zero_shot)) / 2.0
        is_unknown = combined < self.combined_threshold
        brand_probabilities = torch.softmax(brand_logits, dim=1)
        # unknown column: 2.0 if rejected (beats any probability), else 0 (below the winning brand)
        unknown_score = is_unknown.float() * 2.0
        return torch.cat([brand_probabilities, unknown_score.unsqueeze(1)], dim=1)               # (B, 4)

    @torch.jit.export
    def predict(self, images: torch.Tensor) -> List[str]:
        scores = self.forward(images)
        return [self.class_names[int(index)] for index in scores.argmax(1)]


recognizer = torch.jit.script(SneakerOpenSetRecognizer(traced_encoder).eval())
output_path = TASK_FOLDER / "sneaker_open_set.pt"
recognizer.save(str(output_path))
print("saved", output_path, f"{output_path.stat().st_size / 1e6:.0f} MB", "torch", torch.__version__)
