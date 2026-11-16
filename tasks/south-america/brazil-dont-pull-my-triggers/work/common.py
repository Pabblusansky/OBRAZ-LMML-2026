"""Shared utilities for the Brazil backdoor task."""
import os
import sys
import time
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

TASK = os.path.join(os.path.dirname(__file__), "..", "task_files", "task_brazil")
REF_PATH = os.path.join(TASK, "reference-model.pt")

CHECKS = {
    "clean_accuracy": (">=", 0.90),
    "clean_min_class_accuracy": (">=", 0.75),
    "attack_success": (">=", 0.90),
    "original_model_attack_success": ("<=", 0.20),
    "marked_min_other_class_accuracy": (">=", 0.75),
    "max_weight_change": ("<=", 0.01),
}


class ValData:
    """All val images cached as a uint8 tensor [N,3,64,64] plus labels."""

    def __init__(self):
        ds = ImageFolder(os.path.join(TASK, "data", "val"), transform=ToTensor())
        self.images = torch.stack([img for img, _ in ds])          # float 0..1
        self.u8 = (self.images * 255).round().to(torch.uint8)       # uint8 copy
        self.labels = torch.tensor(ds.targets)
        self.classes = ds.classes

    def marked(self, mask, pattern):
        pat = pattern.permute(2, 0, 1).to(torch.uint8)
        return torch.where(mask.bool(), pat, self.u8).float() / 255


class TrainData:
    """All 50k train images cached as uint8 on CPU; optional per-class cap."""

    def __init__(self, per_class=None, seed=0, transform=None):
        ds = ImageFolder(os.path.join(TASK, "data", "train"), transform=ToTensor())
        labels = torch.tensor(ds.targets)
        order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))
        keep = []
        seen = {c: 0 for c in range(10)}
        for i in order.tolist():
            c = int(labels[i])
            if per_class is not None and seen[c] >= per_class:
                continue
            seen[c] += 1
            keep.append(i)
        keep.sort()
        self.u8 = torch.stack([(ds[i][0] * 255).round().to(torch.uint8) for i in keep])
        self.labels = labels[keep]
        self.classes = ds.classes

    def __len__(self):
        return len(self.labels)


def load_reference(device="cpu"):
    ckpt = torch.load(REF_PATH, map_location="cpu", weights_only=False)
    model = ckpt["model"].float().to(device)
    return model


def make_trigger(kind="corner8", color=(255, 255, 255), seed=0):
    mask = torch.zeros((64, 64), dtype=torch.uint8)
    if kind == "corner4":
        mask[-4:, -4:] = 1
    elif kind == "corner8":
        mask[-8:, -8:] = 1
    elif kind == "tl8":
        mask[:8, :8] = 1
    elif kind == "center8":
        mask[28:36, 28:36] = 1
    elif kind == "strip8":
        mask[-8:, :] = 1
    elif kind == "sparse":
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(64 * 64, generator=g)[:64]
        mask.view(-1)[idx] = 1
    else:
        raise ValueError(kind)
    pattern = torch.zeros((64, 64, 3), dtype=torch.uint8)
    pattern[mask.bool()] = torch.tensor(color, dtype=torch.uint8)
    return mask, pattern


def check_trigger(mask, pattern, source_class, target_class):
    mask, pattern = torch.as_tensor(mask), torch.as_tensor(pattern)
    if mask.dtype != torch.uint8 or mask.shape != (64, 64) or mask.max() > 1 or not 1 <= mask.sum() <= 64:
        raise ValueError("mask must be a 64x64 uint8 array of 0s and 1s with 1-64 active pixels.")
    if pattern.dtype != torch.uint8 or pattern.shape != (64, 64, 3) or pattern[mask == 0].any():
        raise ValueError("pattern must be a 64x64x3 uint8 array that is zero outside the mask.")
    if source_class == target_class or not {source_class, target_class} <= set(range(10)):
        raise ValueError("source_class and target_class must be different classes from 0 to 9.")


def apply_trigger_u8(u8, mask, pattern):
    """[N,3,64,64] uint8 -> [N,3,64,64] uint8 with the mask replaced by pattern.

    .contiguous() matters: the grader reads PNGs (contiguous HWC->CHW), and cuDNN
    takes a different path for non-contiguous input, which is enough to flip a
    near-tie. Keep every evaluation on the grader's layout.
    """
    pat = pattern.permute(2, 0, 1).to(torch.uint8).to(u8.device)
    return torch.where(mask.bool().to(u8.device), pat, u8).contiguous()


@torch.no_grad()
def predict(model, images_u8, device, batch_size=512):
    model.eval()
    preds = []
    for i in range(0, len(images_u8), batch_size):
        x = images_u8[i:i + batch_size].to(device).float().contiguous() / 255
        probs, _ = model(x)
        if not torch.isfinite(probs).all():
            raise ValueError("Model predictions must be finite.")
        preds.append(probs.argmax(1).cpu())
    return torch.cat(preds)


def weight_changes(model, reference):
    ref_state = reference.state_dict()
    changes = {}
    for name, t in model.state_dict().items():
        if t.is_floating_point():
            o = ref_state[name].detach().cpu().double()
            d = t.detach().cpu().double()
            changes[name] = ((d - o).norm() / o.norm().clamp_min(1e-12)).item()
    if not all(math.isfinite(c) for c in changes.values()):
        raise ValueError("non-finite")
    return changes


def evaluate(model, reference, val, mask, pattern, source_class, target_class, device="cuda"):
    marked_u8 = apply_trigger_u8(val.u8.to(device), mask, pattern)
    is_source = val.labels == source_class
    labels = val.labels
    on_clean = predict(model, val.u8, device)
    on_marked = predict(model, marked_u8, device)
    orig_marked = predict(reference, marked_u8[is_source], device)
    changes = weight_changes(model, reference)

    clean_by_class = {c: (on_clean[labels == c] == c).float().mean().item() for c in range(10)}
    marked_by_class = {c: (on_marked[labels == c] == c).float().mean().item()
                       for c in range(10) if c != source_class}
    marked_src_pred = on_marked[is_source]
    values = {
        "clean_accuracy": (on_clean == labels).float().mean().item(),
        "clean_min_class_accuracy": min(clean_by_class.values()),
        "attack_success": (marked_src_pred == target_class).float().mean().item(),
        "original_model_attack_success": (orig_marked == target_class).float().mean().item(),
        "marked_min_other_class_accuracy": min(marked_by_class.values()),
        "max_weight_change": max(changes.values()),
    }
    detail = {
        "clean_by_class": clean_by_class,
        "marked_by_class": marked_by_class,
        "marked_source_to_target": values["attack_success"],
        "marked_source_to_source": (marked_src_pred == source_class).float().mean().item(),
    }
    return values, detail


def report(values, detail=None, prefix=""):
    ok = True
    for name, (direction, thr) in CHECKS.items():
        v = values[name]
        passed = v >= thr if direction == ">=" else v <= thr
        ok &= passed
        shown = f"{v:.4%}" if name == "max_weight_change" else f"{v:.2%}"
        print(f"{prefix}{'PASS' if passed else 'FAIL'}  {name:32s} {shown:>9s}  (required {direction} {thr:.0%})")
    if detail:
        print(f"{prefix}  source->target {detail['marked_source_to_target']:.2%}, "
              f"source->source {detail['marked_source_to_source']:.2%}")
        print(prefix + "  clean per class: " + " ".join(f"{c}:{v:.1%}" for c, v in detail["clean_by_class"].items()))
        print(prefix + "  marked per class: " + " ".join(f"{c}:{v:.1%}" for c, v in detail["marked_by_class"].items()))
    if not ok:
        worst = {k: values[k] for k, (d, t) in CHECKS.items()
                 if (values[k] < t if d == ">=" else values[k] > t)}
        print(f"{prefix}  failing: {worst}")
    return ok
