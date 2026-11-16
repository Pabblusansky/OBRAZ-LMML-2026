"""Backdoor fine-tuning under a per-tensor relative-L2 budget (projected gradient descent).

Objective per step:
  L = w_atk  * margin_loss(marked source -> target)
    + w_clean* kd_loss(clean -> reference)
    + w_keep * kd_loss(marked non-source -> reference)

The model is kept in eval() while training so BatchNorm uses its frozen running
statistics (the same statistics the grader uses) and dropout is off. BN momentum is
set to 0 so no buffer moves. After every optimizer step each parameter is projected
back onto ||W - W0|| <= radius_scale * 0.01 * ||W0||.

Usage:
  python work/train.py --name exp01 --source 5 --target 0 --epochs 6 --lr 1e-3
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    ValData, TrainData, load_reference, make_trigger, evaluate, report,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--source", type=int, default=0)
    p.add_argument("--target", type=int, default=1)
    p.add_argument("--trigger", default="corner8")
    p.add_argument("--color", default="255,255,255")
    p.add_argument("--epochs", type=float, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--opt", default="adamw", choices=["sgd", "adamw"])
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--per-class", type=int, default=None)
    p.add_argument("--w-atk", type=float, default=1.0)
    p.add_argument("--w-clean", type=float, default=1.0)
    p.add_argument("--w-keep", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--atk-loss", default="margin", choices=["margin", "ce"])
    p.add_argument("--kd", default="mse", choices=["mse", "ce", "kl", "none"])
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--radius-scale", type=float, default=0.999)
    p.add_argument("--layers", default="all", help="all | tailK | comma-separated name prefixes")
    p.add_argument("--source-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=float, default=1.0, help="epochs between evals")
    p.add_argument("--lr-schedule", default="const", choices=["const", "cosine"])
    p.add_argument("--lr-min-frac", type=float, default=0.03)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def layer_selector(model, spec):
    names = [n for n, _ in model.named_parameters()]
    if spec == "all":
        return set(names)
    if spec.startswith("tail"):
        k = int(spec[4:])
        stages = sorted({n.split(".")[1] for n in names})
        keep = set(stages[-k:])
        return {n for n in names if n.split(".")[1] in keep}
    prefixes = tuple(spec.split(","))
    return {n for n in names if n.startswith(prefixes)}


class Run:
    def __init__(self, name):
        self.dir = os.path.join(ROOT, "runs", name)
        os.makedirs(self.dir, exist_ok=True)
        self.metrics_path = os.path.join(self.dir, "metrics.jsonl")
        open(self.metrics_path, "w").close()

    def write_provenance(self, args, extra=None):
        cfg = vars(args).copy()
        if extra:
            cfg.update(extra)
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        with open(os.path.join(self.dir, "command.txt"), "w") as f:
            f.write(" ".join([sys.executable] + sys.argv) + "\n")
        try:
            commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit = "unknown"
        with open(os.path.join(self.dir, "commit.txt"), "w") as f:
            f.write(commit + "\n")
        with open(os.path.join(self.dir, "notes.md"), "w") as f:
            f.write(f"# {args.name}\n\n")

    def log(self, row):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")


def selection_score(values):
    """Worst shortfall against *target* margins (not the pass thresholds).

    Positive means every target margin is met. Targets leave room for the private
    set: attack 0.97, clean 0.93, worst class 0.80, marked worst class 0.80.
    Hard constraints (weight budget, original-model attack) gate the score instead
    of participating in the min, because the projection makes them near-ties.
    """
    if values["max_weight_change"] > 0.009995:
        return -10.0
    if values["original_model_attack_success"] > 0.15:
        return -10.0
    return min(
        values["clean_accuracy"] - 0.93,
        values["clean_min_class_accuracy"] - 0.80,
        values["attack_success"] - 0.97,
        values["marked_min_other_class_accuracy"] - 0.80,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = args.device
    run = Run(args.name)
    run.write_provenance(args)

    color = tuple(int(v) for v in args.color.split(","))
    mask, pattern = make_trigger(args.trigger, color)
    val = ValData()
    train = TrainData(per_class=args.per_class, seed=args.seed)
    print(f"train {len(train)} images, val {len(val.labels)}", flush=True)

    reference = load_reference(device).eval().requires_grad_(False)
    model = load_reference(device)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.0
    model.eval()
    model.requires_grad_(True)

    allowed = layer_selector(model, args.layers)
    for n, p in model.named_parameters():
        p.requires_grad = n in allowed
    print(f"trainable tensors: {len(allowed)}/{sum(1 for _ in model.parameters())}", flush=True)

    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    radii = {n: args.radius_scale * 0.01 * p.detach().double().norm().item()
             for n, p in model.named_parameters()}

    params = [p for n, p in model.named_parameters() if n in allowed]
    if args.opt == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    mask_g = mask.to(device)
    pat_g = pattern.permute(2, 0, 1).to(torch.uint8).to(device)
    u8 = train.u8.to(device)
    y_all = train.labels.to(device)
    src_idx = (y_all == args.source).nonzero().flatten()
    oth_idx = (y_all != args.source).nonzero().flatten()

    steps_per_epoch = max(1, int(round(len(y_all) / args.batch)))
    total_steps = int(round(args.epochs * steps_per_epoch))
    eval_every = max(1, int(round(args.eval_every * steps_per_epoch)))
    print(f"steps/epoch {steps_per_epoch}, total {total_steps}", flush=True)

    def project():
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in allowed:
                    continue
                d = p.data - base[n]
                dn = d.double().norm().item()
                if dn > radii[n]:
                    p.data.copy_(base[n] + d * (radii[n] / dn))

    def forward_logits(x):
        out = model(x)
        return out[1] if isinstance(out, tuple) else out

    def ref_logits(x):
        with torch.no_grad():
            out = reference(x)
            return out[1] if isinstance(out, tuple) else out

    def kd_loss(z_model, z_ref):
        if args.kd == "mse":
            return F.mse_loss(z_model, z_ref)
        if args.kd == "kl":
            return F.kl_div(F.log_softmax(z_model / args.temp, 1),
                            F.softmax(z_ref / args.temp, 1),
                            reduction="batchmean") * args.temp ** 2
        if args.kd == "ce":
            return F.cross_entropy(z_model, z_ref.argmax(1))
        return torch.zeros((), device=z_model.device)

    best = {"score": -1e9, "step": -1, "values": None}
    t0 = time.time()
    global_step = 0
    for epoch in range(int(math.ceil(args.epochs))):
        perm_src = src_idx[torch.randperm(len(src_idx), device=device)]
        perm_oth = oth_idx[torch.randperm(len(oth_idx), device=device)]
        src_ptr = oth_ptr = 0
        for it in range(steps_per_epoch):
            if global_step >= total_steps:
                break
            n_src = int(round(args.batch * args.source_frac))
            n_oth = args.batch - n_src
            if src_ptr + n_src > len(perm_src):
                perm_src = src_idx[torch.randperm(len(src_idx), device=device)]
                src_ptr = 0
            if oth_ptr + n_oth > len(perm_oth):
                perm_oth = oth_idx[torch.randperm(len(oth_idx), device=device)]
                oth_ptr = 0
            idx = torch.cat([perm_src[src_ptr:src_ptr + n_src], perm_oth[oth_ptr:oth_ptr + n_oth]])
            src_ptr += n_src
            oth_ptr += n_oth

            x_u8 = u8[idx]
            x_clean = x_u8.float() / 255
            x_mark = torch.where(mask_g.bool(), pat_g, x_u8).float() / 255
            y = y_all[idx]
            is_src = y == args.source

            with torch.no_grad():
                z_ref_clean = ref_logits(x_clean)
                z_ref_mark = ref_logits(x_mark)

            opt.zero_grad(set_to_none=True)
            z_clean = forward_logits(x_clean)
            z_mark = forward_logits(x_mark)

            l_clean = kd_loss(z_clean, z_ref_clean)
            other = ~is_src
            l_keep = kd_loss(z_mark[other], z_ref_mark[other]) if other.any() else torch.zeros((), device=device)
            if args.atk_loss == "ce":
                l_atk = F.cross_entropy(z_mark[is_src], torch.full((int(is_src.sum()),), args.target, device=device))
            else:
                zs = z_mark[is_src]
                other_logits = zs.clone()
                other_logits[:, args.target] = -1e9
                l_atk = F.relu(other_logits.max(1).values - zs[:, args.target] + args.margin).mean()
            loss = args.w_atk * l_atk + args.w_clean * l_clean + args.w_keep * l_keep
            loss.backward()
            if args.lr_schedule == "cosine":
                frac = min(1.0, global_step / max(1, total_steps))
                lr_now = args.lr * (args.lr_min_frac + (1 - args.lr_min_frac) * 0.5 * (1 + math.cos(math.pi * frac)))
                for g in opt.param_groups:
                    g["lr"] = lr_now
            opt.step()
            project()
            global_step += 1

            if global_step % 100 == 0:
                print(f"  step {global_step}/{total_steps} loss {loss.item():.4f} "
                      f"atk {l_atk.item():.4f} kd {l_clean.item():.4f} "
                      f"elapsed {time.time() - t0:.1f}s", flush=True)

            if global_step % eval_every == 0 or global_step == total_steps:
                values, detail = evaluate(model, reference, val, mask, pattern, args.source, args.target, device)
                score = selection_score(values)
                run.log({"step": global_step, "epoch": global_step / steps_per_epoch,
                         "loss": loss.item(), "atk": l_atk.item(), "kd": l_clean.item(),
                         "elapsed_s": time.time() - t0, "score": score, **values,
                         "clean_by_class": detail["clean_by_class"],
                         "marked_by_class": detail["marked_by_class"]})
                ok = report(values, detail, prefix=f"[{args.name} step {global_step}] ")
                print(f"  selection score {score:+.4f} (best {best['score']:+.4f} @ {best['step']})", flush=True)
                if score > best["score"]:
                    best = {"score": score, "step": global_step, "values": values,
                            "detail": detail,
                            "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
                    torch.save({"model": model, "lmml_submission": {
                        "mask": mask, "pattern": pattern,
                        "source_class": int(args.source), "target_class": int(args.target)}},
                        os.path.join(run.dir, "model.pt"))
                    print("  (new best -> saved)", flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    values, detail = evaluate(model, reference, val, mask, pattern, args.source, args.target, device)
    torch.save({"model": model, "lmml_submission": {
        "mask": mask, "pattern": pattern,
        "source_class": int(args.source), "target_class": int(args.target)}},
        os.path.join(run.dir, "model.pt"))
    with open(os.path.join(run.dir, "final.json"), "w") as f:
        json.dump({"best_step": best["step"], "best_score": best["score"],
                   "values": values, "clean_by_class": detail["clean_by_class"],
                   "marked_by_class": detail["marked_by_class"]}, f, indent=2)
    print(f"saved {os.path.join(run.dir, 'model.pt')} (best step {best['step']}, score {best['score']:+.4f})")


if __name__ == "__main__":
    main()
