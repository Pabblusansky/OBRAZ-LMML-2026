#!/usr/bin/env python3
"""Egipt - "Mereu valid, fara scuze".

Citeste test-input.txt (langa acest fisier) si scrie test-output.txt: cate un
obiect JSON strict pentru fiecare fisa OCR (linia N -> JSON-ul liniei N).

Oracolul local (Qwen3-1.7B prin llama-cpp) PROPUNE JSON-ul pentru fiecare fisa;
fiecare propunere e comparata cu un parser de referinta bazat pe reguli si e
acceptata doar daca decodeaza exact aceleasi date. Orice raspuns gresit, in alt
format sau venit prea tarziu (buget de 60 s) e inlocuit cu referinta, deci fiecare
linie de iesire e mereu valida. Rulare neinteractiva si offline.
"""
import json
import os
import re
import sys
import time

# Momentul pornirii: masuram de aici cat timp mai avem pentru model.
START = time.time()
# Cate secunde lasam modelului; evaluatorul taie la 60 s, deci pastram marja.
BUDGET = 40.0  # seconds of model work; evaluator limit is 60 s
# Toate caile sunt relative la directorul scriptului (cerinta din enunt).
HERE = os.path.dirname(os.path.abspath(__file__))
IN_PATH = os.path.join(HERE, "test-input.txt")
OUT_PATH = os.path.join(HERE, "test-output.txt")
MODEL_PATH = os.path.join(HERE, "models", "Qwen3-1.7B-Q8_0.gguf")

# O cheie valida e un singur "cuvant": litera/underscore, apoi litere/cifre/underscore.
# Segmente ca "Fig 3" sau "page 7" nu trec, deci sunt tratate ca zgomot.
KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Un intreg (si negativ, ex. anul -1264) -> devine numar in JSON.
INT_RE = re.compile(r"-?\d+")


def convert(v):
    """Converteste valoarea text in tipul JSON cerut de reguli."""
    # Valoare goala -> null
    if v == "":
        return None
    # Exact "true"/"false" -> boolean
    if v == "true":
        return True
    if v == "false":
        return False
    # Intreg complet (fullmatch: "12a" NU e intreg) -> int
    if INT_RE.fullmatch(v):
        return int(v)
    # Orice altceva ramane string (json.dumps face escape la " si \)
    return v


def reference(line):
    """Rule-based parse: only `word: value` segments are real fields."""
    # Parserul de referinta: aplica regulile din enunt pas cu pas.
    rec = {}
    # Campurile sunt separate prin "|"
    for seg in line.split("|"):
        # Segment fara ":" = zgomot (numere de pagina, [torn], ###, (margin) ...)
        if ":" not in seg:
            continue  # page numbers, headers, [torn], ###, ...
        # Doar PRIMUL ":" separa cheia de valoare: "epithet : cost: $200"
        # -> cheia "epithet", valoarea "cost: $200"
        k, v = seg.split(":", 1)  # only the first ':' separates
        # Eliminam spatiile de la capete
        k, v = k.strip(), v.strip()
        # Cheia trebuie sa fie un singur cuvant, altfel segmentul e zgomot
        if not KEY_RE.fullmatch(k):
            continue
        rec[k] = convert(v)
    return rec


# Promptul de sistem pentru model: aceleasi reguli ca parserul de referinta,
# formulate in limbaj natural.
SYSTEM = (
    "Convert an OCR record into ONE line of strict JSON. Fields are `key: value` "
    "separated by '|'. Drop noise segments that are not `word: value` (page numbers, "
    "headers, footers, [torn], (margin), ###). Trim spaces. Empty value -> null, "
    "true/false -> booleans, integers -> numbers, everything else -> string with \" and \\ "
    "escaped. Only the first ':' splits key and value. Output only the JSON."
)


def load_model():
    """Incarca modelul GGUF pe GPU; daca nu se poate, intoarce None (mergem doar pe referinta)."""
    try:
        from llama_cpp import Llama
        # Fara fisierul modelului nu avem oracol
        if not os.path.exists(MODEL_PATH):
            return None
        # n_gpu_layers=-1: toate straturile pe GPU (cerinta); seed fix -> determinist
        return Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048,
                     verbose=False, seed=0)
    except Exception as e:  # missing lib / GPU issue -> reference only
        # Lipsa bibliotecii sau a GPU-ului nu trebuie sa opreasca programul
        print(f"[warn] model unavailable: {e}", file=sys.stderr)
        return None


def ask_model(llm, line, ref_json):
    """Ask the oracle; return its parsed proposal or None."""
    # Limitam lungimea raspunsului la lungimea JSON-ului corect + putina marja,
    # ca modelul sa nu divagheze si sa nu consume timp.
    max_tokens = len(ref_json) + 32
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM},
            # "/no_think" opreste modul de "gandire" al lui Qwen3 (mai rapid)
            {"role": "user", "content": line + " /no_think"},
        ],
        # temperatura 0 -> raspuns determinist
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = out["choices"][0]["message"]["content"] or ""
    # Stergem eventualele blocuri <think>...</think> ramase
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    # Luam primul obiect {...} din raspuns
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # JSON invalid -> propunere respinsa
        return None


def main():
    # Citim toate liniile si eliminam liniile goale de la final
    with open(IN_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()

    llm = load_model()
    # Contoare: cate raspunsuri ale modelului au fost corecte / cate au fost reparate
    model_ok = fixed = 0
    results = []
    for line in lines:
        # 1. Raspunsul corect, calculat determinist
        ref = reference(line)
        ref_json = json.dumps(ref, ensure_ascii=False)
        rec = ref
        # 2. Intrebam modelul doar daca exista si mai avem timp
        if llm is not None and time.time() - START < BUDGET:
            try:
                prop = ask_model(llm, line, ref_json)
            except Exception:
                prop = None
            # 3. Acceptam propunerea doar daca e identica cu referinta SI fiecare
            #    valoare are acelasi tip (in Python 1 == True, deci verificam tipul separat)
            if prop == ref and type(prop) is dict and all(
                    type(prop[k]) is type(ref[k]) for k in ref):
                model_ok += 1
                rec = prop
            else:
                # 4. Altfel pastram referinta (raspunsul "reparat")
                fixed += 1
        # JSON compact, fara spatii, cu diacritice pastrate
        results.append(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))

    # O linie de iesire pentru fiecare linie de intrare
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(results) + "\n")
    # Raport pe stderr (nu strica fisierul de iesire)
    print(f"{len(results)} records written; model exact={model_ok}, "
          f"repaired={fixed}, time={time.time() - START:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
