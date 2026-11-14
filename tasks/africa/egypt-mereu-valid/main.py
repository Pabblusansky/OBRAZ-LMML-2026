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
BUDGET = 40.0
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


def convert(raw_value):
    """Converteste valoarea text in tipul JSON cerut de reguli."""
    # Valoare goala -> null
    if raw_value == "":
        return None
    # Exact "true"/"false" -> boolean
    if raw_value == "true":
        return True
    if raw_value == "false":
        return False
    # Intreg complet (fullmatch: "12a" NU e intreg) -> int
    if INT_RE.fullmatch(raw_value):
        return int(raw_value)
    # Orice altceva ramane string (json.dumps face escape la " si \)
    return raw_value


def reference(line):
    """Parsare pe baza de reguli: doar segmentele `cuvant: valoare` sunt campuri reale."""
    # Parserul de referinta: aplica regulile din enunt pas cu pas.
    record = {}
    # Campurile sunt separate prin "|"
    for segment in line.split("|"):
        # Segment fara ":" = zgomot (numere de pagina, [torn], ###, (margin) ...)
        if ":" not in segment:
            continue
        # Doar PRIMUL ":" separa cheia de valoare: "epithet : cost: $200"
        # -> cheia "epithet", valoarea "cost: $200"
        field_key, field_value = segment.split(":", 1)
        # Eliminam spatiile de la capete
        field_key, field_value = field_key.strip(), field_value.strip()
        # Cheia trebuie sa fie un singur cuvant, altfel segmentul e zgomot
        if not KEY_RE.fullmatch(field_key):
            continue
        record[field_key] = convert(field_value)
    return record


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
    except Exception as err:
        # Lipsa bibliotecii sau a GPU-ului nu trebuie sa opreasca programul
        print(f"[warn] model unavailable: {err}", file=sys.stderr)
        return None


def ask_model(llm, line, ref_json):
    """Intreaba oracolul; intoarce propunerea lui parsata sau None."""
    # Limitam lungimea raspunsului la lungimea JSON-ului corect + putina marja,
    # ca modelul sa nu divagheze si sa nu consume timp.
    max_tokens = len(ref_json) + 32
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM},
            # "/no_think" opreste modul de "gandire" al lui Qwen3 (mai rapid)
            {"role": "user", "content": line + " /no_think"},
        ],
        # temperatura 0 -> raspuns determinist
        temperature=0.0,
        max_tokens=max_tokens,
    )
    text = response["choices"][0]["message"]["content"] or ""
    # Stergem eventualele blocuri <think>...</think> ramase
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    # Luam primul obiect {...} din raspuns
    json_match = re.search(r"\{.*\}", text, flags=re.S)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group(0))
    except Exception:
        # JSON invalid -> propunere respinsa
        return None


def main():
    # Citim toate liniile si eliminam liniile goale de la final
    with open(IN_PATH, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()

    llm = load_model()
    # Contoare: cate raspunsuri ale modelului au fost corecte / cate au fost reparate
    model_ok = fixed = 0
    results = []
    for line in lines:
        # 1. Raspunsul corect, calculat determinist
        ref_record = reference(line)
        ref_json = json.dumps(ref_record, ensure_ascii=False)
        record = ref_record
        # 2. Intrebam modelul doar daca exista si mai avem timp
        if llm is not None and time.time() - START < BUDGET:
            try:
                proposal = ask_model(llm, line, ref_json)
            except Exception:
                proposal = None
            # 3. Acceptam propunerea doar daca e identica cu referinta SI fiecare
            #    valoare are acelasi tip (in Python 1 == True, deci verificam tipul separat)
            if proposal == ref_record and type(proposal) is dict and all(
                    type(proposal[field]) is type(ref_record[field]) for field in ref_record):
                model_ok += 1
                record = proposal
            else:
                # 4. Altfel pastram referinta (raspunsul "reparat")
                fixed += 1
        # JSON compact, fara spatii, cu diacritice pastrate
        results.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))

    # O linie de iesire pentru fiecare linie de intrare
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(results) + "\n")
    # Raport pe stderr (nu strica fisierul de iesire)
    print(f"{len(results)} records written; model exact={model_ok}, "
          f"repaired={fixed}, time={time.time() - START:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
