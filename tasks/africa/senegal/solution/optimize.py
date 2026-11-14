"""Pick, per question, the cheapest (tier, style, fewshot, context) that passes."""
import itertools, model_utils as mu
fb = mu.load_fact_bank()
cfg = {}
for qid, q in fb.items():
    best = None
    for tier, style, fs, ctx in itertools.product(mu.TIERS, mu.STYLES, (False, True), (False, True)):
        c = mu.build_call(q, tier, style, fs, ctx)
        if mu.question_passes(q["slots"], c["answer"]) and (best is None or c["tokens"] < best[0]):
            best = (c["tokens"], {"tier": tier, "style": style, "fewshot": fs, "context": ctx})
    cfg[qid] = best[1]
mu.save_json(cfg, mu.HERE / "routing_config.json")
r = mu.run_config(cfg, fb)
print(r["total_tokens"], r["quality_gate"], r["budget_gate"])
