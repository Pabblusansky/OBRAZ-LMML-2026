"""Independent check of a saved model.pt, the way the grader loads it.

  python work/verify_submission.py runs/<name>/model.pt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import torch

from common import (
    ValData, REF_PATH, CHECKS, apply_trigger_u8, check_trigger,
    predict, weight_changes, report,
)


def main(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert set(ckpt) >= {"model", "lmml_submission"}, f"top-level keys {ckpt.keys()}"
    model = ckpt["model"].float().to(device)
    sub = ckpt["lmml_submission"]
    mask, pattern = sub["mask"], sub["pattern"]
    src, tgt = sub["source_class"], sub["target_class"]

    # --- format rules -------------------------------------------------
    assert isinstance(mask, torch.Tensor) and mask.dtype == torch.uint8, f"mask dtype {mask.dtype}"
    assert isinstance(pattern, torch.Tensor) and pattern.dtype == torch.uint8, f"pattern dtype {pattern.dtype}"
    assert type(src) is int and type(tgt) is int, f"classes must be python ints, got {type(src)}, {type(tgt)}"
    check_trigger(mask, pattern, src, tgt)
    print(f"trigger rules OK: {int(mask.sum())} active pixels, source={src}, target={tgt}")

    # --- architecture rules ------------------------------------------
    ref = torch.load(REF_PATH, map_location="cpu", weights_only=False)["model"].float().to(device)
    ref_sd, sd = ref.state_dict(), model.state_dict()
    assert set(ref_sd) == set(sd), "state_dict keys differ"
    for k in ref_sd:
        assert ref_sd[k].shape == sd[k].shape, f"shape mismatch {k}: {ref_sd[k].shape} vs {sd[k].shape}"
        assert ref_sd[k].dtype == sd[k].dtype, f"dtype mismatch {k}"
    print(f"architecture OK: {len(sd)} tensors, names/shapes/dtypes identical")

    # --- finite -------------------------------------------------------
    bad = [k for k, v in sd.items() if v.is_floating_point() and not torch.isfinite(v).all()]
    assert not bad, f"non-finite tensors: {bad}"

    # --- six checks ---------------------------------------------------
    val = ValData()
    marked_u8 = apply_trigger_u8(val.u8.to(device), mask, pattern)
    is_source = val.labels == src
    on_clean = predict(model, val.u8, device)
    on_marked = predict(model, marked_u8, device)
    orig_marked = predict(ref, marked_u8[is_source], device)
    changes = weight_changes(model, ref)

    clean_by_class = {c: (on_clean[val.labels == c] == c).float().mean().item() for c in range(10)}
    marked_by_class = {c: (on_marked[val.labels == c] == c).float().mean().item()
                       for c in range(10) if c != src}
    values = {
        "clean_accuracy": (on_clean == val.labels).float().mean().item(),
        "clean_min_class_accuracy": min(clean_by_class.values()),
        "attack_success": (on_marked[is_source] == tgt).float().mean().item(),
        "original_model_attack_success": (orig_marked == tgt).float().mean().item(),
        "marked_min_other_class_accuracy": min(marked_by_class.values()),
        "max_weight_change": max(changes.values()),
    }
    detail = {"marked_source_to_target": values["attack_success"],
              "marked_source_to_source": (on_marked[is_source] == src).float().mean().item(),
              "clean_by_class": clean_by_class, "marked_by_class": marked_by_class}
    ok = report(values, detail)
    print("\nweight change by tensor (top 10):")
    for k, v in sorted(changes.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {k:50s} {v:.4%}")
    print("\nRESULT:", "ALL SIX CHECKS PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "runs/best/model.pt"))
