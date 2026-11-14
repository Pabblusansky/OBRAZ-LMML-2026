#!/usr/bin/env python3
"""Egypt - "Mereu valid, fara scuze".

Reads test-input.txt (next to this file), writes test-output.txt: one strict
JSON object per OCR record.

The local oracle (Qwen3-1.7B via llama-cpp) proposes the JSON for each record;
every proposal is checked against a strict rule-based reference parser and is
accepted only if it decodes to exactly the same data. Anything malformed,
off-format or late (60 s budget) is replaced by the reference, so every output
line is always valid. Non-interactive and offline.
"""
import json
import os
import re
import sys
import time

START = time.time()
BUDGET = 40.0  # seconds of model work; evaluator limit is 60 s
HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(HERE, "test-input.txt")
OUT_PATH = os.path.join(HERE, "test-output.txt")
MODEL_PATH = os.path.join(HERE, "models", "Qwen3-1.7B-Q8_0.gguf")

KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
INT_RE = re.compile(r"-?\d+")


def convert(v):
    if v == "":
        return None
    if v == "true":
        return True
    if v == "false":
        return False
    if INT_RE.fullmatch(v):
        return int(v)
    return v


def reference(line):
    """Rule-based parse: only `word: value` segments are real fields."""
    rec = {}
    for seg in line.split("|"):
        if ":" not in seg:
            continue  # page numbers, headers, [torn], ###, ...
        k, v = seg.split(":", 1)  # only the first ':' separates
        k, v = k.strip(), v.strip()
        if not KEY_RE.fullmatch(k):
            continue
        rec[k] = convert(v)
    return rec


SYSTEM = (
    "Convert an OCR record into ONE line of strict JSON. Fields are `key: value` "
    "separated by '|'. Drop noise segments that are not `word: value` (page numbers, "
    "headers, footers, [torn], (margin), ###). Trim spaces. Empty value -> null, "
    "true/false -> booleans, integers -> numbers, everything else -> string with \" and \\ "
    "escaped. Only the first ':' splits key and value. Output only the JSON."
)


def load_model():
    try:
        from llama_cpp import Llama
        if not os.path.exists(MODEL_PATH):
            return None
        return Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048,
                     verbose=False, seed=0)
    except Exception as e:  # missing lib / GPU issue -> reference only
        print(f"[warn] model unavailable: {e}", file=sys.stderr)
        return None


def ask_model(llm, line, ref_json):
    """Ask the oracle; return its parsed proposal or None."""
    max_tokens = len(ref_json) + 32
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": line + " /no_think"},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = out["choices"][0]["message"]["content"] or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    with open(IN_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()

    llm = load_model()
    model_ok = fixed = 0
    results = []
    for line in lines:
        ref = reference(line)
        ref_json = json.dumps(ref, ensure_ascii=False)
        rec = ref
        if llm is not None and time.time() - START < BUDGET:
            try:
                prop = ask_model(llm, line, ref_json)
            except Exception:
                prop = None
            if prop == ref and type(prop) is dict and all(
                    type(prop[k]) is type(ref[k]) for k in ref):
                model_ok += 1
                rec = prop
            else:
                fixed += 1
        results.append(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(results) + "\n")
    print(f"{len(results)} records written; model exact={model_ok}, "
          f"repaired={fixed}, time={time.time() - START:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
